"""Generic HTML -> structured SourcePart extraction with independent recovery.

The original HTML is immutable. Blocks and inline runs are the source of both
the canonical Markdown and the visual preview. Image descriptions are never
invented; unknown images retain their URL, position and attachment.
"""
from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse
from . import fallback

STRUCTURED_IMPORT_ERROR = None
try:
    from bs4 import BeautifulSoup, Comment, NavigableString, Tag
    from bs4.builder._html5lib import HTML5TreeBuilder
    import cssselect2
    import tinycss2
except ImportError as exc:
    STRUCTURED_IMPORT_ERROR = str(exc)

VERSION = "html-structured-prototype.v12"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes() + b'\0fallback.py\0' +
                                      Path(fallback.__file__).read_bytes()).hexdigest()
HEADINGS = {f"h{i}" for i in range(1, 7)}
SKIP = {"script", "style", "noscript", "template"}
EMBEDDED = {'iframe', 'frame', 'embed', 'object'}
INLINE_CONTAINER = {'p', 'span', 'a', 'b', 'strong', 'i', 'em', 'code', 'sub', 'sup', 'label', 'td', 'th', 'li'}
BLOCK = HEADINGS | {"p", "div", "section", "article", "main", "body", "ul", "ol",
                    "li", "dl", "dt", "dd", "blockquote", "pre", "table", "hr",
                    "header", "footer", "nav", "figure", "figcaption", "details", "summary"}


def authored_visual(node):
    """Keep safe visual values that themselves carry information (e.g. colors)."""
    values = {}
    for declaration in str(node.get('style', '')).split(';'):
        if ':' in declaration:
            key, value = declaration.split(':', 1)
            key, value = key.strip().lower(), value.strip()
            if key in {'background', 'background-color', 'color'} and re.fullmatch(r'[#\w\s(),.%+-]+', value):
                values[key] = value
    for attribute, key in [('bgcolor', 'background-color'), ('color', 'color')]:
        value = str(node.get(attribute, ''))
        if value and re.fullmatch(r'[#\w\s(),.%+-]+', value):
            values[key] = value
    return values


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8", errors='surrogatepass') if isinstance(value, str) else value).hexdigest()


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", "")).strip()


def parse_html(value: str):
    return BeautifulSoup(value, builder=HTML5TreeBuilder())


def content_root(soup):
    """Keep all sibling main/article regions rather than taking only the first."""
    def common_owner(nodes):
        owners = [{id(n), *(id(p) for p in n.parents)} for n in nodes]
        for candidate in [nodes[0], *nodes[0].parents]:
            if all(id(candidate) in group for group in owners):
                return candidate
        return soup.body or soup
    for candidates in [soup.find_all(lambda n: isinstance(n, Tag) and (n.name == 'main' or n.get('role') == 'main')),
                       soup.find_all('article')]:
        ids = {id(n) for n in candidates}
        outer = [n for n in candidates if not any(id(p) in ids for p in n.parents)]
        if outer:
            return common_owner(outer)
    return soup.body or soup


def content_navigation(node):
    return (node.get('role') == 'doc-toc' or node.name == 'nav' and
            bool(re.search(r'\b(?:table of contents|contents)\b', str(node.get('aria-label', '')), re.I)))


def ignored_region(node):
    if not isinstance(node, Tag):
        return False
    return node.name in SKIP or node.name == 'footer' or (
        (node.name == 'nav' or 'nav' in node.get('class', [])) and not content_navigation(node))


def is_block_code(node):
    """Recognize explicit block-code markup without knowing its publisher.

    Syntax highlighting does not necessarily use <pre>. Never guess from the
    programming language or prose content; use structural author declarations.
    """
    if not isinstance(node, Tag):
        return False
    if node.name == 'pre':
        return True
    if node.name != 'code':
        return False
    classes = {c.casefold() for c in node.get('class', [])}
    style = str(node.get('style', '')).casefold()
    return bool(classes & {'block', 'code-block', 'codeblock'} or
                re.search(r'(?:^|;)\s*display\s*:\s*block\b', style) or
                re.search(r'(?:^|;)\s*white-space\s*:\s*pre(?:-wrap)?\b', style))


def authored_code_language(node):
    """Retain an explicit, unambiguous language declaration; never infer one."""
    declarations = []
    for owner in (node, *node.find_all('code')):
        language = owner.get('data-language')
        if isinstance(language, str) and re.fullmatch(r'[A-Za-z0-9_+.-]{1,40}', language):
            declarations.append(language)
        for value in owner.get('class', []):
            match = re.fullmatch(r'(?:language|lang)-([A-Za-z0-9_+.-]{1,40})', str(value))
            if match:
                declarations.append(match[1])
    unique = list(dict.fromkeys(declarations))
    return unique[0] if len(unique) == 1 else None


def plain(runs: list[dict], image_marker: bool = True) -> str:
    return "".join("\n" if r["type"] == "break" else
                   f"[image {r['image_id']}]" if r["type"] == "image" and image_marker else
                   "" if r["type"] == "image" else
                   (embedded_text(r['resource']) if image_marker else '') + plain(r.get('fallback_runs', []), image_marker)
                   if r['type'] == 'embedded' else r.get("text", "") for r in runs).strip()


def captured_attributes(node):
    return {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in node.attrs.items()}


def embedded_text(resource):
    refs = ' | '.join(r['attribute'] + ': ' + r['reference'] for r in resource['references'])
    state = {'external_content_not_captured': '内部内容尚未捕获', 'inline_document_saved': '内联 HTML 已保存',
             'no_source_declared': '未声明外部来源'}[resource['content_status']]
    text = ('[嵌入资源 ' + resource['embedded_id'] + ' · ' + resource['tag'] +
            (' | ' + refs if refs else '') + ' | ' + state + ']')
    if resource.get('inline_html'):
        text += '\n\n' + fenced_code(resource['inline_html']['content'])
    return text


class Extractor:
    def __init__(self, source_url: str, source_id: str = "SRC1", resource_base_url: str | None = None,
                 stylesheets: list[str] | None = None, resource_urls: dict[str, str] | None = None,
                 limits: fallback.Limits | None = None):
        self.url = source_url
        self.source_id = source_id
        self.resource_base_url = resource_base_url
        self.resource_urls = resource_urls or {}
        self.stylesheets = stylesheets or []
        self.limits = limits or fallback.Limits()
        self.hidden_nodes = {}
        self.layout = {}
        self.annotations = {}
        self.recording_annotation = False
        self.expanding_hidden = set()
        self.links: dict[str, dict] = {}
        self.images: dict[str, dict] = {}
        self.embedded: dict[str, dict] = {}
        self.embedded_paths: dict[str, str] = {}
        self.link_ids: dict[str, str] = {}
        self.paths: dict[int, str] = {}
        self.warnings: list[dict] = []
        self._source_soup = self._source_root = None

    def initialize_visibility(self, soup):
        """Resolve static author display rules; retain hidden inline material.

        Conditional media queries and interactive states are not flattened into
        the default view. No publisher-specific classes are known to this code.
        """
        winners = {}
        properties = {'display', 'float', 'width', 'height', 'max-width'}
        # BeautifulSoup excludes Stylesheet text from get_text(); the literal
        # contents are the CSS source, not a document text node.
        sheets = self.stylesheets + [''.join(str(c) for c in n.contents) for n in soup.find_all('style')]
        order = 0
        matcher = cssselect2.Matcher()
        for sheet in sheets:
            for rule in tinycss2.parse_stylesheet(sheet, skip_comments=True, skip_whitespace=True):
                if rule.type != 'qualified-rule':
                    continue
                displays = [d for d in tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
                            if d.type == 'declaration' and d.lower_name in properties]
                if not displays:
                    continue
                try:
                    selectors = cssselect2.compile_selector_list(rule.prelude)
                except cssselect2.SelectorError:
                    continue
                declarations = []
                for declaration in displays:
                    order += 1
                    declarations.append((declaration.lower_name,
                                         tinycss2.serialize(declaration.value).strip().casefold(),
                                         int(declaration.important), order))
                for selector in selectors:
                    if selector.pseudo_element:
                        continue
                    matcher.add_selector(selector, declarations)
        # Matching each selector through soup.select repeatedly scanned the
        # entire large document. cssselect2 indexes selectors by tag/class/id
        # and tests only relevant rules while traversing the DOM once.
        owners = {}
        def selector_tree(node):
            tag = '{' + node.namespace + '}' + node.name if node.namespace else node.name
            element = ET.Element(tag, {str(k): ' '.join(v) if isinstance(v, list) else str(v or '')
                                       for k, v in node.attrs.items()})
            owners[id(element)] = node
            previous = None
            for child in node.contents:
                if isinstance(child, Tag):
                    previous = selector_tree(child)
                    element.append(previous)
                elif isinstance(child, NavigableString) and not isinstance(child, Comment):
                    if previous is None:
                        element.text = (element.text or '') + str(child)
                    else:
                        previous.tail = (previous.tail or '') + str(child)
            return element
        html_root = soup.html or soup.find(True)
        if html_root is not None:
            wrapper = cssselect2.ElementWrapper.from_html_root(selector_tree(html_root))
            for element in wrapper.iter_subtree():
                node = owners[id(element.etree_element)]
                for specificity, selector_order, pseudo, declarations in matcher.match(element):
                    for prop, value, important, source_order in declarations:
                        rank = (important, 0, *specificity, source_order)
                        key = (id(node), prop)
                        if rank >= winners.get(key, ((-1,), None))[0]:
                            winners[key] = (rank, value)
        for node in soup.find_all(True):
            inline_style = node.get('style')
            declarations = tinycss2.parse_declaration_list(str(inline_style), skip_comments=True, skip_whitespace=True) if inline_style else []
            for declaration in declarations:
                if declaration.type == 'declaration' and declaration.lower_name in properties:
                    rank = (int(declaration.important), 1, 0, 0, 0, order + 1)
                    key = (id(node), declaration.lower_name)
                    if rank >= winners.get(key, ((-1,), None))[0]:
                        winners[key] = (rank, tinycss2.serialize(declaration.value).strip().casefold())
            self.layout[id(node)] = {prop: winners[(id(node), prop)][1] for prop in properties
                                     if (id(node), prop) in winners}
            if self.layout[id(node)].get('display') == 'none':
                self.hidden_nodes[id(node)] = 'author_display_none'
            elif node.has_attr('hidden') and (id(node), 'display') not in winners:
                self.hidden_nodes[id(node)] = 'author_hidden_attribute'

    def capture_annotation(self, node, styles, link_id, preserve):
        key = 'A' + str(len(self.annotations) + 1)
        self.recording_annotation = True
        try:
            if node.name == 'img':
                image_run = self.image(node)
                image_run['link_id'] = link_id
                runs = [image_run]
            elif node.name in EMBEDDED:
                runs = self.inline([node], styles, link_id, preserve, True)
            else:
                owned_link = self.link(node) if node.name == 'a' else link_id
                runs = self.inline(node.contents, styles, owned_link, preserve, True)
        finally:
            self.recording_annotation = False
        self.annotations[key] = {'annotation_id': key, 'dom_path': self.path(node),
                                 'reason': self.hidden_nodes[id(node)], 'context': 'code' if preserve else 'inline',
                                 'runs': runs, 'raw_html': str(node), 'raw_sha256': digest(str(node))}

    def index_paths(self, node: Tag, path: str = "document") -> None:
        self.paths[id(node)] = path
        counts: Counter = Counter()
        text_index = 0
        for child in node.contents:
            if isinstance(child, Tag):
                counts[child.name] += 1
                self.index_paths(child, f"{path}/{child.name}[{counts[child.name]}]")
            else:
                text_index += 1
                self.paths[id(child)] = f"{path}/text()[{text_index}]"

    def path(self, node) -> str:
        return self.paths.get(id(node), "document")

    def link(self, node: Tag) -> str | None:
        if not node.has_attr('href'):
            return None
        href = str(node.get("href") or "").strip()
        url = urljoin(self.base, href)
        scheme = urlparse(url).scheme
        clickable = scheme in {"https", "http", "file", "mailto"}
        if url not in self.link_ids:
            key = f"L{len(self.links) + 1}"
            self.link_ids[url] = key
            self.links[key] = {"link_id": key, "url": url, "href": href,
                               "occurrences": [], "labels": [], "clickable": clickable,
                               "kind": "web" if clickable else "application_command" if scheme == 'cxcmd' else "non_web"}
        key = self.link_ids[url]
        label = compact(node.get_text(" ", strip=True))
        self.links[key]["occurrences"].append(self.path(node))
        if label and label not in self.links[key]["labels"]:
            self.links[key]["labels"].append(label)
        return key

    def image(self, node: Tag) -> dict:
        key = f"I{len(self.images) + 1}"
        src = str(node.get("src") or "")
        owner = node.find_parent(["p", "li", "td", "th", "figure"])
        context = "table_cell" if owner and owner.name in {"td", "th"} else (
            "inline" if owner and owner.name != "figure" and compact(owner.get_text()) else "standalone")
        self.images[key] = {
            "image_id": key, "src": src, "url": self.resource_urls.get(src, urljoin(self.resource_base_url or self.base, src)),
            "alt": str(node.get("alt") or ""), "title": str(node.get("title") or ""),
            "dom_path": self.path(node), "role": "unknown", "status": "pending",
            "display_context": context, "author_classes": node.get("class", []),
            "author_width": node.get("width"), "author_height": node.get("height"),
            "author_style": node.get('style', ''),
            "author_layout": self.layout.get(id(node), {}),
        }
        return {"type": "image", "image_id": key, "dom_path": self.path(node)}

    def register_embedded(self, node):
        """Capture a container/reference, without loading or executing its body.

        Attributes plus the original snapshot preserve the source declaration.
        Never serialize whole nested subtrees into every resource record.
        """
        path = self.path(node)
        if path in self.embedded_paths:
            return self.embedded[self.embedded_paths[path]]
        key = 'EM' + str(len(self.embedded) + 1)
        attributes = captured_attributes(node)
        inline_html = None
        if node.name == 'iframe' and node.has_attr('srcdoc'):
            value = str(node.get('srcdoc') or '')
            inline_html = {'content': value, 'char_count': len(value), 'sha256': digest(value), 'format': 'html'}
        references = []
        for attribute in ('src', 'data'):
            value = str(node.get(attribute) or '')
            if value:
                url = self.resource_urls.get(value, fallback.resolve(self.resource_base_url or self.base, value))
                references.append({'attribute': attribute, 'reference': value, 'url': url,
                                   'clickable': fallback.clickable(url), 'content_captured': False,
                                   'active': inline_html is None})
        status = 'inline_document_saved' if inline_html is not None else 'external_content_not_captured' if references else 'no_source_declared'
        resource = {'embedded_id': key, 'tag': node.name, 'dom_path': path, 'attributes': attributes,
                    'references': references, 'inline_html': inline_html, 'content_status': status,
                    'synthetic_label': True}
        self.embedded_paths[path] = key
        self.embedded[key] = resource
        return resource

    def inline(self, nodes, styles: tuple[str, ...] = (), link_id: str | None = None,
               preserve: bool = False, nested: bool = False) -> list[dict]:
        """Collapse HTML whitespace, never insert separators at inline tags.

        Normalization happens across the whole run sequence so ``+<b>Shift</b>``
        stays ``+Shift`` and literal spaces across tags remain single spaces.
        """
        result: list[dict] = []
        for node in nodes:
            if isinstance(node, Comment):
                continue
            if isinstance(node, NavigableString):
                text = str(node).replace("\x00", "")
                if not preserve:
                    text = re.sub(r"\s+", " ", text)
                if text:
                    visual = {}
                    for ancestor in reversed(list(node.parents)):
                        if isinstance(ancestor, Tag) and ancestor.name not in (BLOCK | {'tr', 'td', 'th', 'tbody', 'thead', 'tfoot'}):
                            visual.update(authored_visual(ancestor))
                    result.append({"type": "text", "text": text,
                                   "styles": list(styles), "link_id": link_id,
                                   "visual": visual,
                                   "dom_path": self.path(node)})
            elif isinstance(node, Tag) and node.name not in SKIP:
                if id(node) in self.hidden_nodes and not self.recording_annotation:
                    self.capture_annotation(node, styles, link_id, preserve)
                    continue
                if node.name == "br":
                    result.append({"type": "break", "dom_path": self.path(node)})
                elif node.name == "img":
                    run = self.image(node)
                    run["link_id"] = link_id
                    result.append(run)
                elif node.name in EMBEDDED:
                    resource = self.register_embedded(node)
                    result.append({'type': 'embedded', 'embedded_id': resource['embedded_id'], 'resource': resource,
                                   'fallback_runs': self.inline(node.contents, styles, link_id, preserve, True),
                                   'styles': [], 'link_id': link_id, 'dom_path': self.path(node)})
                else:
                    next_styles = styles
                    style = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic",
                             "code": "code", "sup": "sup", "sub": "sub"}.get(node.name)
                    if style and style not in styles:
                        next_styles = (*styles, style)
                    next_link = self.link(node) if node.name == "a" else link_id
                    result.extend(self.inline(node.contents, next_styles, next_link, preserve, True))
        if preserve or nested:
            return result
        previous_space = True
        for run in result:
            if run["type"] == "text":
                text = run["text"]
                if previous_space:
                    text = text.lstrip(" ")
                run["text"] = text
                if text:
                    previous_space = text.endswith(" ")
            else:
                if run["type"] == "break":
                    for prev in reversed(result[:result.index(run)]):
                        if prev["type"] == "text" and prev["text"]:
                            prev["text"] = prev["text"].rstrip(" ")
                            break
                previous_space = run["type"] == "break"
        for run in reversed(result):
            if run["type"] == "text":
                run["text"] = run["text"].rstrip(" ")
                if run["text"]:
                    break
            else:
                break
        return [r for r in result if r["type"] != "text" or r["text"]]

    def block(self, kind: str, node, **kwargs) -> dict:
        return {"type": kind, "dom_path": self.path(node), **kwargs}

    def sequence(self, nodes) -> list[dict]:
        result: list[dict] = []
        pending: list = []

        def flush():
            if pending:
                runs = self.inline(pending)
                if runs and (plain(runs) or any(r["type"] == "image" for r in runs)):
                    text_runs = [r for r in runs if r["type"] == "text" and r["text"].strip()]
                    owner = pending[0] if isinstance(pending[0], Tag) else pending[0].parent
                    command_context = owner and (owner.name == "blockquote" or owner.find_parent("blockquote"))
                    in_table = owner and (owner.name in {"td", "th"} or owner.find_parent(["td", "th"]))
                    command = bool(command_context and not in_table and len(plain(runs).split()) > 1 and text_runs and
                                   all("bold" in r["styles"] or "code" in r["styles"] for r in text_runs))
                    # The inline siblings belong to their shared container.
                    # Using the first text node as block owner would orphan a
                    # hidden sibling's annotation from its SourcePart.
                    chains = [[n, *n.parents] for n in pending]
                    memberships = [{id(n) for n in chain} for chain in chains]
                    container = next((n for n in chains[0] if isinstance(n, Tag) and
                                      all(id(n) in group for group in memberships)), owner)
                    result.append(self.block("command" if command else "paragraph", container, runs=runs))
                pending.clear()

        for node in nodes:
            if isinstance(node, Comment):
                continue
            if not isinstance(node, Tag):
                pending.append(node)
                continue
            if node.name in SKIP:
                continue
            if ignored_region(node):
                flush()
                continue
            # Inline wrappers may surround list/block elements in older manuals.
            contains_blocks = node.find(list(BLOCK | (EMBEDDED if node.name not in INLINE_CONTAINER else set()))) is not None
            if (id(node) in self.hidden_nodes and id(node) not in self.expanding_hidden
                    and (node.name in BLOCK or node.name in EMBEDDED or contains_blocks or is_block_code(node))):
                flush()
                self.expanding_hidden.add(id(node))
                try:
                    children = self.sequence([node])
                finally:
                    self.expanding_hidden.remove(id(node))
                result.append(self.block('hidden_content', node, children=children,
                                         visibility_reason=self.hidden_nodes[id(node)]))
                continue
            if node.name in EMBEDDED:
                if node.parent and node.parent.name in INLINE_CONTAINER:
                    pending.append(node)
                    continue
                flush()
                resource = self.register_embedded(node)
                result.append(self.block('embedded_content', node, embedded_id=resource['embedded_id'],
                                         resource=resource, children=self.sequence(node.contents)))
                continue
            if node.name not in BLOCK and not contains_blocks and not is_block_code(node):
                pending.append(node)
                continue
            flush()
            if node.name in HEADINGS:
                runs = self.inline(node.contents)
                # A leading back-arrow link is page chrome, not part of the title.
                runs = [r for r in runs if not (r.get("link_id") and
                        compact(r.get("text", "")) in {"←", "¶", "#"})]
                result.append(self.block("heading", node, level=int(node.name[1]), runs=runs))
            elif is_block_code(node):
                result.append(self.block("code", node, runs=self.inline(node.contents, preserve=True),
                                         author_tag=node.name, author_classes=node.get('class', []),
                                         language=authored_code_language(node)))
            elif node.name in {"ul", "ol"}:
                items = []
                def list_children(container):
                    for child in container.contents:
                        if isinstance(child, Tag) and child.name == 'li':
                            yield self.block('list_item', child, children=self.sequence(child.contents))
                        elif isinstance(child, Tag) and child.name not in BLOCK and child.find('li'):
                            yield from list_children(child)
                        elif isinstance(child, Tag) or (isinstance(child, NavigableString) and str(child).strip()):
                            yield self.block('list_content', child, children=self.sequence([child]))
                items.extend(list_children(node))
                result.append(self.block("list", node, ordered=node.name == "ol",
                                         start=int(node.get("start", 1)),
                                         marker="none" if "none" in node.get("class", []) else "normal",
                                         items=items))
            elif node.name == 'details':
                summary = node.find('summary', recursive=False)
                children = self.sequence([x for x in node.contents if x is not summary])
                result.append(self.block('disclosure', node,
                    runs=self.inline(summary.contents) if summary else [],
                    open=node.has_attr('open'), anchor_id=node.get('id'), children=children))
            elif node.name == "table":
                result.append(self.table(node))
            elif node.name == 'figure':
                result.append(self.block('figure', node, anchor_id=node.get('id'), children=self.sequence(node.contents)))
            elif node.name == 'dl':
                entries, pending_entry = [], []
                for child in self.sequence(node.contents):
                    if child['type'] == 'definition_term' and any(c['type'] == 'definition' for c in pending_entry):
                        entries.append(self.block('definition_item', node, children=pending_entry))
                        pending_entry = []
                    pending_entry.append(child)
                if pending_entry:
                    entries.append(self.block('definition_item', node, children=pending_entry))
                result.append(self.block('definition_list', node, items=entries))
            elif node.name == "blockquote":
                result.append(self.block("quote", node, children=self.sequence(node.contents)))
            elif node.name in {"dt", "dd"}:
                result.append(self.block("definition_term" if node.name == "dt" else "definition",
                                         node, children=self.sequence(node.contents)))
            elif node.name in {"figcaption", "summary"}:
                result.append(self.block("caption", node, runs=self.inline(node.contents)))
            elif node.name != "hr":
                children = self.sequence(node.contents)
                # Publishers may split an example into several adjacent code
                # elements (e.g. interleaved source and output). Their shared
                # wrapper is meaningful; keep it without publisher selectors.
                if node.name not in {'p', 'body', 'main', 'article'} and len(children) > 1 and all(c['type'] in {'code', 'code_group'} for c in children):
                    result.append(self.block('code_group', node, children=children))
                else:
                    result.extend(children)
        flush()
        return result

    def table(self, node: Tag) -> dict:
        rows = []
        occupied: dict[tuple[int, int], str] = {}
        source_rows = [r for r in node.find_all('tr') if r.find_parent('table') is node]
        for row_index, tr in enumerate(source_rows):
            cells = []
            col = 0
            for cell in tr.find_all(["td", "th"], recursive=False):
                while (row_index, col) in occupied:
                    col += 1
                rowspan = max(1, int(cell.get("rowspan", 1)))
                colspan = max(1, int(cell.get("colspan", 1)))
                key = f"r{row_index + 1}c{col + 1}"
                content = self.sequence(cell.contents)
                cells.append(self.block("cell", cell, cell_id=key, column=col,
                                        rowspan=rowspan, colspan=colspan,
                                        header=cell.name == "th", scope=cell.get("scope"),
                                        visual=authored_visual(cell), title=cell.get('title'),
                                        author_width=cell.get('width'), author_height=cell.get('height'),
                                        children=content))
                for rr in range(row_index, row_index + rowspan):
                    for cc in range(col, col + colspan):
                        occupied[rr, cc] = key
                col += colspan
            rows.append(self.block("row", tr, row_index=row_index, cells=cells, visual=authored_visual(tr)))
        first = node.find("tr")
        headers = bool(first and first.find_all("th", recursive=False))
        reason = "authored_th" if headers else None
        if not headers and first:
            first_cells = first.find_all(["td", "th"], recursive=False)
            headers = bool(first_cells and all(c.find(["b", "strong"]) and
                           len(compact(c.get_text())) <= 65 and not c.find("img") and
                           not c.get("rowspan") and not c.get("colspan") for c in first_cells))
            reason = "short_bold_first_row" if headers else None
        if headers and rows:
            for cell in rows[0]["cells"]:
                cell["header"] = True
        caption = node.find("caption", recursive=False)
        return self.block("table", node, rows=rows, header_row=headers,
                          author_layout=self.layout.get(id(node), {}),
                          author_classes=node.get('class', []), author_align=node.get('align'),
                          caption_runs=self.inline(caption.contents) if caption else [],
                          header_detection=reason, column_count=max((cc + 1 for rr, cc in occupied), default=0))

    def extract(self, html_text: str, max_chars: int = 10_000) -> dict:
        """Return SourceParts for captured HTML, with independent recovery.

        Cancellation and process termination remain control signals. Invalid
        API parameters remain caller errors rather than fabricated captures.
        """
        if not isinstance(html_text, str):
            raise TypeError('html_text must be decoded HTML text')
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
            raise ValueError('max_chars must be a positive integer')
        # A reused extractor must not retain IDs or partially built structures
        # from its previous page, including a failed structured attempt.
        self.__init__(self.url, self.source_id, self.resource_base_url,
                      self.stylesheets, self.resource_urls, self.limits)
        probe, reasons = fallback.scan(html_text, self.url, self.limits,
                                       self.resource_base_url, self.resource_urls)
        reasons = list(reasons)
        if STRUCTURED_IMPORT_ERROR:
            reasons.append({'code': 'structured_dependencies_unavailable', 'message': STRUCTURED_IMPORT_ERROR})
        if not reasons:
            try:
                document = self._extract_structured(html_text, max_chars)
                reasons = self._quality_issues(document, probe)
                if not reasons:
                    document['source_snapshot'] = fallback.source_snapshot(html_text)
                    document['extraction'] = {'mode': 'structured', 'degraded': False, 'reasons': [],
                                              'capture_status': 'saved', 'coverage': 'selected_content_root',
                                              'embedded_sources_pending': any(e['content_status'] == 'external_content_not_captured'
                                                                              for e in document['embedded_resources']),
                                              'render_required': bool(not probe.content_chars and document['embedded_resources'])}
                    for resource in document['embedded_resources']:
                        resource['source_snapshot_sha256'] = document['raw_sha256']
                    for part in document['parts']:
                        part['metadata']['extraction_mode'] = 'structured'
                        part['metadata']['degraded'] = False
                        part['metadata']['source_snapshot_sha256'] = document['raw_sha256']
                    return document
            except Exception as exc:
                reasons = [{'code': 'structured_extraction_failed', 'error_type': type(exc).__name__,
                            'message': str(exc)[:400]}]
        # Discard contaminated DOM/annotation state. Recovery does not call
        # parse_html, sequence, CSS matching, Markdown or make_parts.
        self.__init__(self.url, self.source_id, self.resource_base_url,
                      self.stylesheets, self.resource_urls, self.limits)
        return fallback.recover(html_text, self.url, self.source_id, max_chars, VERSION,
                                IMPLEMENTATION_SHA256, reasons, probe, self.limits.fallback_parts)

    def _quality_issues(self, document, probe):
        if not document.get('parts'):
            return [{'code': 'structured_parts_empty'}]
        root = self._source_root
        unsupported = sorted({n.name for n in root.find_all(['svg', 'math', 'canvas', 'picture'])
                              if not any(ignored_region(p) for p in n.parents)})
        if unsupported:
            return [{'code': 'unrepresented_structured_elements', 'elements': unsupported}]
        expected = {}
        for node in root.descendants:
            if not isinstance(node, NavigableString) or isinstance(node, Comment) or not compact(str(node)):
                continue
            ancestors = list(node.parents)
            if any(ignored_region(n) for n in ancestors):
                continue
            if (compact(str(node)) in {'←', '¶', '#'} and any(n.name in HEADINGS for n in ancestors)
                    and any(n.name == 'a' and n.has_attr('href') for n in ancestors)):
                continue
            expected[self.path(node)] = compact(str(node))
        runs = [r for section in document['sections'] for r in [*inline_runs(section['heading_runs']), *all_runs(section['blocks'])]]
        runs.extend(r for a in document['annotations'].values() for r in inline_runs(a['runs']))
        actual = Counter(r['dom_path'] for r in runs if r['type'] == 'text' and compact(r['text']) and r['dom_path'] in expected)
        missing = sum(not actual[p] for p in expected)
        duplicated = sum(n != 1 for n in actual.values())
        altered = sum(compact(r['text']) != expected[r['dom_path']] for r in runs
                      if r['type'] == 'text' and r['dom_path'] in expected)
        if missing or duplicated or altered:
            return [{'code': 'structured_text_coverage_failed', 'missing': missing, 'duplicated': duplicated, 'altered': altered}]
        raw_images = [n for n in root.find_all('img') if not any(ignored_region(p) for p in n.parents)]
        if any(not n.get('src') for n in raw_images):
            return [{'code': 'structured_image_source_missing'}]
        if {self.path(n) for n in raw_images} - {n['dom_path'] for n in document['images'].values()}:
            return [{'code': 'structured_image_coverage_failed'}]
        raw_links = [n for n in root.find_all('a', href=True) if
                     not any(ignored_region(p) for p in n.parents) and (compact(n.get_text()) or n.find('img'))]
        link_paths = {p for link in document['links'].values() for p in link['occurrences']}
        if {self.path(n) for n in raw_links} - link_paths:
            return [{'code': 'structured_link_coverage_failed'}]
        raw_embedded = [n for n in root.find_all(list(EMBEDDED)) if not any(ignored_region(p) for p in [n, *n.parents])]
        saved_embedded = document['embedded_resources']
        if Counter(self.path(n) for n in raw_embedded) != Counter(e['dom_path'] for e in saved_embedded):
            return [{'code': 'structured_embedded_coverage_failed'}]
        by_path = {e['dom_path']: e for e in saved_embedded}
        if any(by_path[self.path(n)]['attributes'] != captured_attributes(n) for n in raw_embedded):
            return [{'code': 'structured_embedded_attributes_changed'}]
        embedded_ids = {e['embedded_id'] for e in saved_embedded}
        if ({key for p in document['parts'] for key in p['metadata']['embedded_ids']} != embedded_ids or
                any(not set(p['metadata']['embedded_ids']) <= embedded_ids for p in document['parts'])):
            return [{'code': 'structured_embedded_part_coverage_failed'}]
        if any(p['content_sha256'] != digest(p['content']) or p['char_count'] != len(p['content']) for p in document['parts']):
            return [{'code': 'structured_part_integrity_failed'}]
        expected_blocks = [b for s in document['sections'] for b in s['blocks']]
        saved_blocks = [b for p in document['parts'] for b in p['metadata']['blocks']]
        if (saved_blocks != expected_blocks or
                any(p['content'] != markdown_blocks(p['metadata']['blocks']) for p in document['parts']) or
                Counter(r['dom_path'] for s in document['sections'] for r in s['heading_runs']) !=
                Counter(r['dom_path'] for p in document['parts'] for r in p['metadata']['heading_runs'])):
            return [{'code': 'structured_part_coverage_failed'}]
        if any(not (set(p['metadata']['image_ids']) <= set(document['images']) and
                    set(p['metadata']['link_ids']) <= set(document['links'])) for p in document['parts']):
            return [{'code': 'structured_part_references_invalid'}]
        # Empty/wrong main landmarks must not silently hide actual body prose.
        # Ignore navigation/header/aside/footer chrome when checking material
        # outside the selected boundary. Recovery then captures the whole page.
        if root is not self._source_soup.body:
            selected_chars = sum(len(value) for value in expected.values())
            outside_chars = 0
            if selected_chars < 256 and self._source_soup.body:
                for node in self._source_soup.body.descendants:
                    if not isinstance(node, NavigableString) or isinstance(node, Comment):
                        continue
                    ancestors = list(node.parents)
                    if root in ancestors or any(ignored_region(n) or n.name in {'aside', 'header'} for n in ancestors):
                        continue
                    if any(n.name in {'p', 'pre', 'table', 'dl', 'figure', 'section', 'article'} for n in ancestors):
                        outside_chars += len(compact(str(node)))
                if outside_chars > max(64, selected_chars * 2):
                    return [{'code': 'selected_content_root_incomplete', 'selected_chars': selected_chars, 'outside_chars': outside_chars}]
        body_runs = [*all_runs([b for s in document['sections'] for b in s['blocks']]),
                     *(r for a in document['annotations'].values() for r in inline_runs(a['runs']))]
        if not any(r['type'] == 'image' or r['type'] == 'text' and compact(r['text']) for r in body_runs) and probe and probe.content_chars:
            heading_chars = sum(len(compact(r.get('text', ''))) for section in document['sections'] for r in section['heading_runs'])
            if probe.content_chars > heading_chars:
                return [{'code': 'structured_body_empty'}]
        return []

    def _extract_structured(self, html_text: str, max_chars: int = 10_000) -> dict:
        soup = parse_html(html_text)
        self._source_soup = soup
        self.base = urljoin(self.url, str(soup.find("base").get("href"))) if soup.find("base", href=True) else self.url
        self.index_paths(soup)
        self.initialize_visibility(soup)
        title = compact(soup.title.get_text()) if soup.title else self.url
        root = content_root(soup)
        self._source_root = root
        blocks = self.sequence(root.contents)
        # Unclosed paragraphs can put headings inside quotations. Lift them while
        # retaining the surrounding quotation portions and their DOM provenance.
        def lift(items):
            lifted = []
            for block in items:
                if block["type"] == "quote":
                    children = lift(block["children"])
                    group = []
                    for child in children:
                        if child["type"] == "heading":
                            if group:
                                lifted.append({**block, "children": group}); group = []
                            lifted.append(child)
                        else:
                            group.append(child)
                    if group:
                        lifted.append({**block, "children": group})
                else:
                    lifted.append(block)
            return lifted
        blocks = lift(blocks)
        sections = []
        current = {"title": title, "level": 0, "heading_path": [], "heading_runs": [], "blocks": []}
        headings = []
        for block in blocks:
            if block["type"] == "heading":
                if current["blocks"] or current.get('heading_runs'):
                    sections.append(current)
                level = block["level"]
                while headings and headings[-1][0] >= level:
                    headings.pop()
                label = compact(plain(block["runs"], image_marker=False))
                headings.append((level, label))
                current = {"title": label, "level": level,
                           "heading_path": [x[1] for x in headings],
                           "heading_runs": block['runs'],
                           "heading_dom_path": block["dom_path"], "blocks": []}
            else:
                current["blocks"].append(block)
        if current["blocks"] or current.get('heading_runs'):
            sections.append(current)
        result = {"schema_version": VERSION, "source_id": self.source_id, "url": self.url,
                  "implementation_sha256": IMPLEMENTATION_SHA256,
                  "title": title, "parser": "html5lib 1.1", "raw_sha256": digest(html_text), "sections": sections,
                  "content_root": {"tag": root.name, "role": root.get('role'), "dom_path": self.path(root)},
                  "links": self.links, "images": self.images, "warnings": self.warnings}
        result['annotations'] = self.annotations
        result['embedded_resources'] = list(self.embedded.values())
        result['style_context'] = {'external_stylesheets': len(self.stylesheets), 'mode': 'static_author_display_rules'}
        result["parts"] = self.make_parts(result, max_chars)
        return result

    def make_parts(self, document: dict, max_chars: int) -> list[dict]:
        parts = []
        for section_index, section in enumerate(document["sections"], 1):
            groups, pending = [], []
            pending_size = 0
            pending_rendered = 0
            for block in section["blocks"]:
                # Soft limit: complete code/list/table blocks stay intact. A table
                # that exceeds it is explicitly recorded, never cut mid-cell.
                block_text = markdown_blocks([block])
                size = pending_size + len(block_text) + (2 if block_text and pending_rendered else 0)
                if pending and size > max_chars:
                    groups.append(pending); pending = []
                    pending_size, pending_rendered = 0, 0
                    size = len(block_text)
                pending.append(block)
                pending_size = size
                pending_rendered += bool(block_text)
            if pending:
                groups.append(pending)
            if not groups and section.get('heading_runs'):
                groups = [[]]
            for chunk, blocks in enumerate(groups, 1):
                content = markdown_blocks(blocks)
                label = section["title"] + (f" ({chunk})" if len(groups) > 1 else "")
                part_id = "part_" + digest(f"{self.source_id}|{self.url}|{section_index}|{chunk}|{VERSION}")[:20]
                ranges = []
                cursor = 0
                rendered_count = 0
                for block in blocks:
                    text = markdown_blocks([block])
                    if text and rendered_count:
                        cursor += 2
                    ranges.append({"dom_path": block["dom_path"], "block_type": block["type"],
                                   "start_offset": cursor, "end_offset": cursor + len(text)})
                    cursor += len(text)
                    rendered_count += bool(text)
                owned_runs = [*all_runs(blocks), *inline_runs(section.get('heading_runs', []) if chunk == 1 else [])]
                owned_paths = [b['dom_path'] for b in blocks]
                if chunk == 1 and section.get('heading_dom_path'):
                    owned_paths.append(section['heading_dom_path'])
                annotations = [a for a in self.annotations.values() if any(a['dom_path'] == p or a['dom_path'].startswith(p + '/') for p in owned_paths)]
                owned_runs.extend(r for a in annotations for r in inline_runs(a['runs']))
                image_ids = sorted({run["image_id"] for run in owned_runs if run["type"] == "image"}, key=lambda x: int(x[1:]))
                link_ids = sorted({run["link_id"] for run in owned_runs if run.get("link_id")}, key=lambda x: int(x[1:]))
                embedded_ids = sorted({r['embedded_id'] for r in owned_runs if r['type'] == 'embedded'} |
                                      {b['embedded_id'] for b in all_blocks(blocks) if b['type'] == 'embedded_content'})
                parts.append({
                    "part_id": part_id, "source_id": self.source_id, "kind": "html_section" if blocks else "html_heading",
                    "label": label, "locator": {"url": self.url, "section_index": section_index,
                        "heading": section["title"], "heading_level": section["level"],
                        "heading_path": section["heading_path"], "chunk": chunk},
                    "ordinal": section_index * 10_000 + chunk,
                    "content": content, "content_sha256": digest(content), "char_count": len(content),
                    "metadata": {"normalization_version": VERSION, "format": "markdown",
                        "blocks": blocks, "block_ranges": ranges, "link_ids": link_ids,
                        "image_ids": image_ids, "embedded_ids": embedded_ids, "soft_limit_exceeded": len(content) > max_chars},
                })
                parts[-1]['metadata']['heading_runs'] = section.get('heading_runs', []) if chunk == 1 else []
                parts[-1]['metadata']['annotations'] = annotations
                parts[-1]['metadata']['structural_heading'] = not bool(blocks)
        return parts


def inline_runs(runs):
    for run in runs:
        yield run
        yield from inline_runs(run.get('fallback_runs', []))


def all_runs(blocks):
    for block in blocks:
        yield from inline_runs(block.get("runs", []))
        yield from inline_runs(block.get("caption_runs", []))
        yield from all_runs(block.get("children", []))
        yield from all_runs(block.get("items", []))
        yield from all_runs(block.get("rows", []))
        yield from all_runs(block.get("cells", []))


def all_blocks(blocks):
    for block in blocks:
        yield block
        for key in ("children", "items", "rows", "cells"):
            yield from all_blocks(block.get(key, []))


def markdown_runs(runs, code=False):
    runs = [{**r, 'text': embedded_text(r['resource']) +
             (' ' + markdown_runs(r['fallback_runs'], code) if r.get('fallback_runs') else '')}
            if r['type'] == 'embedded' else r for r in runs]
    if not code:
        from itertools import groupby
        fragments = []
        for (link_id, styles), group in groupby(runs, key=lambda r: (r.get('link_id'), tuple(r.get('styles', [])))):
            group = list(group)
            text = ''.join('<br>' if r['type'] == 'break' else f"[image {r['image_id']}]" if r['type'] == 'image'
                           else r.get('text', '') for r in group)
            lead = text[:len(text)-len(text.lstrip(' '))]
            tail = text[len(text.rstrip(' ')):]
            value = text.strip(' ')
            if not value:
                fragments.append((link_id, text))
                continue
            if 'code' in styles:
                fence = '`' * (max((len(s) for s in re.findall(r'`+', value)), default=0) + 1)
                pad = ' ' if value.startswith('`') or value.endswith('`') else ''
                value = fence + pad + value + pad + fence
            else:
                for style in styles:
                    if style in {'sup', 'sub'}:
                        value = f'<{style}>' + value + f'</{style}>'
                    elif style == 'bold':
                        value = '**' + value + '**'
                    elif style == 'italic':
                        value = '*' + value + '*'
            fragments.append((link_id, lead + value + tail))
        output = []
        for link_id, group in groupby(fragments, key=lambda x: x[0]):
            value = ''.join(x[1] for x in group)
            if link_id and value.strip():
                lead = value[:len(value)-len(value.lstrip(' '))]
                tail = value[len(value.rstrip(' ')):]
                value = lead + '[' + value.strip() + '][' + link_id + ']' + tail
            output.append(value)
        return ''.join(output).strip()
    result = []
    index = 0
    while index < len(runs):
        run = runs[index]
        index += 1
        if run["type"] == "break":
            result.append("\n" if code else "<br>")
        elif run["type"] == "image":
            result.append(f"[image {run['image_id']}]")
        else:
            text = run["text"]
            if run.get("link_id") and not code and text.strip():
                while index < len(runs) and runs[index]["type"] == "text" and runs[index].get("link_id") == run["link_id"]:
                    text += runs[index]["text"]
                    index += 1
                lead = " " if text.startswith(" ") else ""
                tail = " " if text.endswith(" ") else ""
                text = f"{lead}[{text.strip()}][{run['link_id']}]{tail}"
            result.append(text)
    return "".join(result) if code else "".join(result).strip()


def fenced_code(text, language=None):
    fence = '`' * max(3, max((len(s) for s in re.findall(r'`+', text)), default=0) + 1)
    info = language if isinstance(language, str) and re.fullmatch(r'[A-Za-z0-9_+.-]{1,40}', language) else ''
    return fence + info + '\n' + text + '\n' + fence


def _table_requires_block_cells(rows):
    """Pipe tables cannot preserve code, nested blocks or paragraph boundaries."""
    for row in rows:
        for cell in row['cells']:
            children = cell['children']
            if len(children) > 1 or any(child['type'] not in {'paragraph', 'caption'} for child in children):
                return True
    return False


def markdown_blocks(blocks, depth=0):
    result = []
    for block in blocks:
        kind = block["type"]
        if kind in {"paragraph", "caption", "heading"}:
            result.append(markdown_runs(block["runs"]))
        elif kind in {'fallback_text', 'fallback_html'}:
            result.append(''.join(r.get('text', '') for r in block['runs']))
        elif kind == 'embedded_content':
            result.append(embedded_text(block['resource']) +
                          ('\n\n' + markdown_blocks(block['children'], depth) if block['children'] else ''))
        elif kind in {"code", "command"}:
            result.append(fenced_code(markdown_runs(block["runs"], code=True), block.get('language')))
        elif kind in {"quote", "definition", "definition_term", "definition_item", "list_item", "list_content", "code_group", "figure"}:
            result.append(markdown_blocks(block["children"], depth))
        elif kind == 'definition_list':
            result.append(markdown_blocks(block['items'], depth))
        elif kind == 'hidden_content':
            result.append('<details>\n<summary>Content hidden in the original HTML</summary>\n\n' +
                          markdown_blocks(block['children'], depth) + '\n\n</details>')
        elif kind == 'disclosure':
            result.append('### ' + markdown_runs(block['runs']) + '\n\n' + markdown_blocks(block['children'], depth))
        elif kind == "list":
            # A hidden marker is presentation, not evidence that prose is code.
            # Markdown uses ordinary list syntax; author marker style stays in
            # the block metadata and the visual view.
            items = []
            for index, item in enumerate(block["items"], block["start"]):
                marker = '' if item['type'] == 'list_content' else f"{index}. " if block["ordered"] else "- "
                text = markdown_blocks(item["children"], depth + 1)
                items.append(marker + text.replace("\n", "\n  "))
            result.append("\n".join(items))
        elif kind == "table":
            if block.get("caption_runs"):
                result.append(markdown_runs(block["caption_runs"]))
            rows = block["rows"]
            def cell_text(cell):
                text = markdown_blocks(cell['children'])
                if not text.strip() and cell.get('visual'):
                    color = cell['visual'].get('background-color') or cell['visual'].get('background')
                    if color:
                        text = '[color swatch: ' + color + ']'
                return text
            simple = all(c["rowspan"] == c["colspan"] == 1 for r in rows for c in r["cells"])
            block_cells = _table_requires_block_cells(rows)
            if simple and block["header_row"] and not block_cells:
                lines = []
                for index, row in enumerate(rows):
                    texts = [cell_text(c).replace("\n", "<br>").replace("|", "\\|") for c in row["cells"]]
                    lines.append("| " + " | ".join(texts) + " |")
                    if index == 0:
                        lines.append("| " + " | ".join("---" for c in row["cells"]) + " |")
                result.append("\n".join(lines))
            else:
                lines = ["[Table — cell positions and spans preserved]"]
                for row in rows:
                    lines.append(f"Row {row['row_index'] + 1}:")
                    for cell in row["cells"]:
                        span = f"; rowspan={cell['rowspan']}; colspan={cell['colspan']}"
                        text = cell_text(cell)
                        if block_cells:
                            # Cell coordinates carry the table relationship;
                            # authored blocks remain at their original indentation.
                            # Indenting or flattening a fence would alter code bytes.
                            lines.append(f"\nColumn {cell['column'] + 1}{span}:\n\n{text}\n")
                        else:
                            lines.append(f"  Column {cell['column'] + 1}{span}: " + text.replace("\n", " / "))
                result.append("\n".join(lines))
    return "\n\n".join(x for x in result if x.strip())


def verify_markdown_code_integrity(blocks, rendered_text):
    """Recheck authored code against final Markdown, including list containers.

    Source/projection hashes cannot detect a later table renderer flattening a
    code block. This independent fence reader removes only the two spaces that
    each enclosing list adds to continuation lines. Original code indentation,
    blank lines, pipes, and trailing newlines remain part of the comparison.
    The checker is opt-in for verification/reporting, not an extraction gate.
    """
    specs = []

    def collect(items, list_depth=0):
        for block in items:
            if block['type'] in {'code', 'command'}:
                text = markdown_runs(block['runs'], code=True)
                fenced = fenced_code(text, block.get('language'))
                opening = fenced.split('\n', 1)[0]
                fence = re.match(r'`+', opening).group()
                specs.append({'text': text, 'opening': opening, 'fence': fence,
                              'list_depth': list_depth, 'dom_path': block.get('dom_path')})
                continue
            nested_depth = list_depth + (block['type'] == 'list')
            for key in ('children', 'items', 'rows', 'cells'):
                collect(block.get(key, []), nested_depth)

    collect(blocks)
    # Split only on LF: authored CR bytes inside a code run must not be silently
    # normalized by Python's broader splitlines() behavior.
    lines = rendered_text.split('\n')
    cursor, checks, errors = 0, [], []
    prefix_pattern = r'(?:(?:[-+*]|\d+[.)])[ \t]+|[ \t])*'
    for index, spec in enumerate(specs, 1):
        pattern = re.compile(r'^' + prefix_pattern + re.escape(spec['opening']) + r'$')
        indent = '  ' * spec['list_depth']
        found = None
        first_mismatch = None
        scan = cursor
        while scan < len(lines):
            if not pattern.fullmatch(lines[scan]):
                scan += 1
                continue
            end = scan + 1
            closing = indent + spec['fence']
            while end < len(lines) and lines[end] != closing:
                end += 1
            if end == len(lines):
                first_mismatch = first_mismatch or 'Code closing fence is missing or has changed container indentation'
                scan += 1
                continue
            body_lines = lines[scan + 1:end]
            if any(not line.startswith(indent) for line in body_lines):
                first_mismatch = first_mismatch or 'Code container indentation changed'
                scan += 1
                continue
            body = '\n'.join(line[len(indent):] for line in body_lines)
            if body != spec['text']:
                first_mismatch = first_mismatch or 'Final Markdown code differs from authored code runs'
                # A closing fence from an unrelated rendered construct can
                # look like an opening fence. Its apparent end may actually be
                # this authored block's real opening: never skip that line.
                scan += 1
                continue
            found = (scan, end, body)
            break
        if found is None:
            errors.append({'code_index': index, 'dom_path': spec['dom_path'],
                           'error': first_mismatch or 'Authored code fence is missing from final Markdown'})
            continue
        start, end, body = found
        cursor = end + 1
        checks.append({'code_index': index, 'dom_path': spec['dom_path'], 'list_depth': spec['list_depth'],
                       'start_line': start + 1, 'end_line': end + 1,
                       'authored_code_sha256': digest(spec['text']), 'final_code_sha256': digest(body)})
    return {'status': 'failed' if errors else 'passed', 'code_blocks_expected': len(specs),
            'code_blocks_checked': len(checks), 'error_count': len(errors), 'errors': errors, 'checks': checks}


def html_embedded(resource, children='', inline=False):
    tag = 'span' if inline else 'div'
    title = str(resource['attributes'].get('title') or '')
    label = '嵌入资源 ' + resource['embedded_id'] + ' · ' + resource['tag'] + (' · ' + title if title else '')
    state = {'external_content_not_captured': '内部内容尚未捕获', 'inline_document_saved': '内联 HTML 已保存',
             'no_source_declared': '未声明外部来源'}[resource['content_status']]
    refs = []
    for ref in resource['references']:
        value = html.escape(ref['reference'])
        if ref['clickable'] and not inline:
            value = '<a href="' + html.escape(ref['url'], quote=True) + '" target="_blank" rel="noopener">' + value + '</a>'
        refs.append(html.escape(ref['attribute']) + ': ' + value)
    markup = '<' + tag + ' class="source-embedded" data-embedded-id="' + resource['embedded_id'] + '"><strong>' + html.escape(label) + '</strong> · ' + state
    if refs:
        markup += '<br>' + '<br>'.join(refs)
    if resource.get('inline_html'):
        source = html.escape(resource['inline_html']['content'])
        markup += ('<br><code>' + source + '</code>') if inline else '<details><summary>查看内联 HTML</summary><pre><code>' + source + '</code></pre></details>'
    if children:
        markup += '<br>' + children
    return markup + '</' + tag + '>'


def html_runs(runs, document, asset_prefix=""):
    output = []
    for run in runs:
        if run["type"] == "break":
            value = "<br>"
        elif run['type'] == 'embedded':
            value = html_embedded(run['resource'], html_runs(run['fallback_runs'], document, asset_prefix), inline=True)
        elif run["type"] == "image":
            img = document["images"][run["image_id"]]
            url = asset_prefix + img["local_path"] if img.get("local_path") else img["url"]
            dimensions = []
            for axis in ("width", "height"):
                authored = str(img.get("author_" + axis) or "")
                if re.fullmatch(r"\d+(?:px|%)?", authored):
                    dimensions.append(f"{axis}:{authored if authored.endswith(('px', '%')) else authored + 'px'}")
            style = ';'.join(dimensions)
            if img.get('author_width') and not img.get('author_height'):
                style += ';height:auto'
            for axis in ('width', 'height', 'max-width'):
                value = img.get('author_layout', {}).get(axis, '')
                if re.fullmatch(r'(?:\d+(?:\.\d+)?(?:px|em|rem|%|vw|vh)|auto|none)', value):
                    dimensions.append(axis + ':' + value)
            floating = img.get('author_layout', {}).get('float')
            if floating in {'left', 'right'}:
                dimensions.append('float:' + floating)
            style = ';'.join(dimensions)
            inline = img.get('display_context') == 'inline'
            image_class = 'source-image inline-image' if inline else 'source-image illustration'
            value = f'<img class="{image_class}" style="{style}" src="{html.escape(url, quote=True)}" alt="{html.escape(img["alt"], quote=True)}" title="{run["image_id"]}: {html.escape(img["url"], quote=True)}">'
            if img.get('status') in {'failed', 'missing'}:
                value = '<span class="missing-image" title="' + html.escape(img['url'], quote=True) + '">[图片资源不可用：' + html.escape(img['alt'] or run['image_id']) + ']</span>'
        else:
            value = html.escape(run["text"])
            if run.get('visual'):
                style = ';'.join(k + ':' + v for k, v in run['visual'].items())
                value = '<span style="' + html.escape(style, quote=True) + '">' + value + '</span>'
            for style in run.get("styles", []):
                tag = {"bold": "strong", "italic": "em", "code": "code", "sup": "sup", "sub": "sub"}[style]
                value = f"<{tag}>{value}</{tag}>"
        output.append((run.get("link_id"), value))
    from itertools import groupby
    rendered = []
    for link_id, group in groupby(output, key=lambda pair: pair[0]):
        value = ''.join(pair[1] for pair in group)
        if link_id:
            url = document["links"][link_id]["url"]
            if document['links'][link_id].get('clickable', True):
                value = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{value}</a>'
            else:
                value = f'<span class="command-link" title="{html.escape(url, quote=True)}">{value}</span>'
        rendered.append(value)
    return ''.join(rendered)


def html_blocks(blocks, document, asset_prefix=""):
    output = []
    for block in blocks:
        kind = block["type"]
        if kind == 'heading':
            level = block['level']
            output.append(f'<h{level}>' + html_runs(block['runs'], document, asset_prefix) + f'</h{level}>')
        elif kind in {"paragraph", "caption"}:
            output.append("<p>" + html_runs(block["runs"], document, asset_prefix) + "</p>")
        elif kind in {"code", "command"}:
            # HTML strips a newline immediately following <pre>. A child code
            # wrapper preserves even leading/blank-only source lines exactly.
            output.append("<pre><code>" + html_runs(block["runs"], document, asset_prefix) + "</code></pre>")
        elif kind in {'fallback_text', 'fallback_html'}:
            output.append('<pre class="source-fallback"><code>' +
                          html.escape(''.join(r.get('text', '') for r in block['runs'])) + '</code></pre>')
        elif kind == 'embedded_content':
            output.append(html_embedded(block['resource'], html_blocks(block['children'], document, asset_prefix)))
        elif kind == "quote":
            output.append("<blockquote>" + html_blocks(block["children"], document, asset_prefix) + "</blockquote>")
        elif kind == 'code_group':
            output.append('<div class="code-group">' + html_blocks(block['children'], document, asset_prefix) + '</div>')
        elif kind == 'hidden_content':
            output.append('<details class="source-hidden"><summary>原网页隐藏的补充内容</summary>' +
                          html_blocks(block['children'], document, asset_prefix) + '</details>')
        elif kind == 'figure':
            contents = []
            for child in block['children']:
                if child['type'] == 'caption':
                    contents.append('<figcaption>' + html_runs(child['runs'], document, asset_prefix) + '</figcaption>')
                else:
                    contents.append(html_blocks([child], document, asset_prefix))
            output.append('<figure>' + ''.join(contents) + '</figure>')
        elif kind == 'definition_list':
            output.append('<dl>' + html_blocks(block['items'], document, asset_prefix) + '</dl>')
        elif kind == 'definition_term':
            output.append('<dt>' + html_blocks(block['children'], document, asset_prefix) + '</dt>')
        elif kind == 'definition':
            output.append('<dd>' + html_blocks(block['children'], document, asset_prefix) + '</dd>')
        elif kind in {"list_item", "definition_item", "list_content"}:
            output.append(html_blocks(block["children"], document, asset_prefix))
        elif kind == 'disclosure':
            output.append('<details open><summary>' + html_runs(block['runs'], document, asset_prefix) + '</summary>' + html_blocks(block['children'], document, asset_prefix) + '</details>')
        elif kind == "list":
            tag = "ol" if block["ordered"] else "ul"
            attr = ' class="unmarked-list" style="list-style:none;padding-left:0"' if block["marker"] == "none" else ""
            if block["ordered"]:
                attr += f' start="{block["start"]}"'
            contents = []
            for item in block['items']:
                contents.append(('<li>' if item['type'] == 'list_item' else '<li class="list-content">') + html_blocks(item['children'], document, asset_prefix) + '</li>')
            output.append(f"<{tag}{attr}>" + ''.join(contents) + f"</{tag}>")
        elif kind == "table":
            rows = []
            for row in block["rows"]:
                cells = []
                for cell in row["cells"]:
                    tag = "th" if cell["header"] else "td"
                    visual = {**row.get('visual', {}), **cell.get('visual', {})}
                    for axis in ['width', 'height']:
                        authored = str(cell.get('author_' + axis) or '')
                        if re.fullmatch(r'\d+(?:px|%)?', authored):
                            visual[axis] = authored if authored.endswith(('px', '%')) else authored + 'px'
                    style = ';'.join(k + ':' + v for k, v in visual.items())
                    title = html.escape(str(cell.get('title') or ''), quote=True)
                    cells.append(f'<{tag} rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}" style="{html.escape(style, quote=True)}" title="{title}">' + html_blocks(cell["children"], document, asset_prefix) + f"</{tag}>")
                row_style = ';'.join(k + ':' + v for k, v in row.get('visual', {}).items())
                rows.append('<tr style="' + html.escape(row_style, quote=True) + '">' + "".join(cells) + "</tr>")
            caption = '<caption>' + html_runs(block["caption_runs"], document, asset_prefix) + '</caption>' if block.get("caption_runs") else ''
            authored_float = block.get('author_layout', {}).get('float') or block.get('author_align')
            floating = authored_float in {'right', 'left'}
            orientation = 'left' if authored_float == 'left' else 'right'
            wrapper = 'table-scroll presentation-' + orientation if floating else 'table-scroll'
            output.append(f'<div class="{wrapper}"><table>' + caption + "".join(rows) + "</table></div>")
    return "\n".join(output)
