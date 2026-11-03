"""Lightweight authored outlines and exact DOM boundaries for selected chapters.

Outline discovery is deliberately local: it never acquires child pages or creates
SourceParts. Captured HTML remains immutable; selection filters the structured
projection before it can become Evidence.
"""
from __future__ import annotations

import copy
import hashlib
import re
from urllib.parse import unquote, urldefrag, urljoin, urlparse

from bs4 import NavigableString, Tag, UnicodeDammit

from ..html_navigation import classify_html_link, html_base_url, is_authored_contents, is_heading_permalink
from .extractor import Extractor, all_blocks, all_runs, compact, content_root, inline_runs, parse_html
from .fallback import Limits, scan


HEADINGS = [f'h{i}' for i in range(1, 7)]
LISTS = ['ul', 'ol', 'dl']


def page_url(value):
    return urldefrag(str(value or ''))[0]


def _label(node):
    if not isinstance(node, Tag):
        return ''
    return compact(' '.join(str(value) for value in node.descendants
                            if isinstance(value, NavigableString)
                            and not (isinstance(value.parent, Tag) and value.parent.name == 'a'
                                     and is_heading_permalink(value.parent))))


def _chrome(node):
    return any(isinstance(owner, Tag) and (
        owner.name in {'script', 'style', 'noscript', 'template', 'header', 'footer'}
        or owner.name == 'nav' and not is_authored_contents(owner)
    ) for owner in [node, *node.parents])


def _identity(source_id, url, title, position):
    value = f'{source_id}|{url}|{title}|{position}'
    return 'htmlsec_' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]


def flatten_sections(sections):
    for section in sections:
        yield section
        yield from flatten_sections(section.get('children', []))


def preview_html_outline(data, *, root_url=None, source_id):
    """Identify selectable first-level chapters using only the captured root."""
    root_url = root_url or f'artifact://{source_id}/root'
    html = (UnicodeDammit(data, is_html=True).unicode_markup or data.decode('utf-8', errors='replace')) if isinstance(data, bytes) else str(data)
    # The outline must not defeat the same complexity guard used by extraction.
    _probe, reasons = scan(html, root_url, Limits())
    if any('budget' in reason['code'] for reason in reasons):
        return {'source_id':source_id, 'modality':'html', 'title':root_url,
                'outline_available':False, 'sections':[],
                'warnings':['The root HTML exceeds the lightweight outline parsing budget.']}
    soup = parse_html(html)
    root = content_root(soup)
    base = html_base_url(soup, root_url)
    title = _label(soup.title) or _label(root.find(HEADINGS)) or root_url
    indexer = Extractor(root_url, source_id)
    indexer.index_paths(soup)

    def link_target(anchor):
        if not isinstance(anchor, Tag) or is_heading_permalink(anchor):
            return None
        href = str(anchor.get('href') or '').strip()
        absolute = urljoin(base, href)
        if not href or urlparse(absolute).scheme not in {'http', 'https', 'file', 'artifact'}:
            return None
        origin, candidate = urlparse(root_url), urlparse(absolute)
        if origin.scheme in {'http', 'https', 'file'} and (
                origin.scheme != candidate.scheme or origin.netloc.casefold() != candidate.netloc.casefold()):
            return None
        role, _reason = classify_html_link(anchor)
        if role not in {'directory_entry', 'local_anchor'}:
            return None
        url, fragment = urldefrag(absolute)
        if url == page_url(root_url) and not fragment:
            return None
        return {'url':url, 'fragment':unquote(fragment) or None}

    def list_items(owner):
        entries = []
        pending = None
        children = owner.find_all(['dt', 'dd'] if owner.name == 'dl' else ['li'], recursive=False)
        for item in children:
            if item.name == 'dd':
                if pending is not None:
                    for nested in item.find_all(LISTS):
                        if nested.find_parent(LISTS) is owner:
                            pending['children'].extend(list_items(nested))
                continue
            direct_anchor = next((a for a in item.find_all('a', href=True)
                                  if a.find_parent(LISTS) is owner and link_target(a)), None)
            label = _label(direct_anchor) if direct_anchor else compact(' '.join(
                str(value) for value in item.find_all(string=True)
                if value.find_parent(LISTS) is owner))
            entry = {'title':label, **(link_target(direct_anchor) or {}), 'children':[]}
            for nested in item.find_all(LISTS):
                if nested.find_parent(LISTS) is owner:
                    entry['children'].extend(list_items(nested))
            if label and (entry.get('url') or entry['children']):
                entries.append(entry)
                pending = entry
        return entries

    # Prefer explicit author-declared contents even when it lives beside <main>.
    candidates = []
    for owner in soup.find_all(LISTS):
        if owner.find_parent(LISTS) or _chrome(owner):
            continue
        entries = list_items(owner)
        if not entries:
            continue
        authored = is_authored_contents(owner)
        # Ordinary unordered lists of prose links are not document outlines.
        if not authored and not any(entry['children'] for entry in entries):
            previous = owner.find_previous(HEADINGS)
            context = _label(previous).casefold()
            if not re.search(r'\b(contents|chapters|manual|handbook|guide)\b|目录', context):
                continue
        candidates.append((authored, sum(1 for _ in flatten_sections(entries)), entries))
    sections = max(candidates, key=lambda value:(value[0], value[1]))[2] if candidates else []
    # Some site generators wrap the document title around its chapter list.
    while len(sections) == 1 and sections[0]['children'] and (
            sections[0]['title'].casefold() == title.casefold()
            or not sections[0].get('url')):
        sections = sections[0]['children']

    if not sections:
        headings = [h for h in root.find_all(HEADINGS) if not _chrome(h) and not is_authored_contents(h) and _label(h)]
        if headings:
            level = min(int(h.name[1]) for h in headings)
            top = [h for h in headings if int(h.name[1]) == level]
            if len(top) == 1 and any(int(h.name[1]) > level for h in headings):
                level = min(int(h.name[1]) for h in headings if int(h.name[1]) > level)
                top = [h for h in headings if int(h.name[1]) == level]
            sections = [{'title':_label(h), 'url':page_url(root_url),
                         'fragment':str(h.get('id') or '') or None,
                         'heading_dom_path':indexer.path(h), 'children':[]} for h in top]

    def finalize(nodes, prefix=''):
        result = []
        for ordinal, node in enumerate(nodes, 1):
            position = f'{prefix}.{ordinal}'
            node = {**node, 'section_id':_identity(source_id, str(node.get('url', '')) + '#' + str(node.get('fragment') or ''),
                                                  node['title'], position)}
            node['children'] = finalize(node.get('children', []), position)
            node['known_page_count'] = len({page_url(item['url']) for item in flatten_sections([node]) if item.get('url')})
            result.append(node)
        return result
    sections = finalize(sections)
    return {'source_id':source_id, 'modality':'html', 'title':title,
            'outline_available':bool(sections), 'sections':sections,
            'warnings':[] if sections else ['No reliable chapter outline was found in the root HTML.']}


def selected_sections(scope):
    """Validate selection against the immutable preview, never arbitrary URLs."""
    sections = scope.get('sections') or []
    requested = scope.get('section_ids') or []
    known = {str(item['section_id']):item for item in sections}
    if not requested or any(str(key) not in known for key in requested):
        raise ValueError('HTML selection must contain known first-level section IDs')
    return [known[str(key)] for key in dict.fromkeys(requested)]


def selected_anchor_urls(soup, current_url, targets):
    """Return directory links whose authored anchor is inside selected bounds."""
    html = str(soup)
    parsed = parse_html(html)
    indexer = Extractor(current_url, 'scope_links')
    indexer.index_paths(parsed)
    base = html_base_url(parsed, current_url)
    anchors = [anchor for anchor in parsed.find_all('a', href=True)
               if classify_html_link(anchor)[0] == 'directory_entry']
    document = {'source_id':'scope_links', 'url':current_url, 'extraction':{'mode':'structured'},
                'sections':[{'title':'links', 'heading_runs':[], 'blocks':[
                    {'type':'paragraph', 'dom_path':indexer.path(anchor),
                     'runs':[{'type':'text', 'text':urljoin(base, str(anchor['href'])),
                              'dom_path':indexer.path(anchor)}]} for anchor in anchors]}],
                'parts':[], 'warnings':[], 'images':{}, 'links':{}, 'annotations':{}, 'embedded_resources':[]}
    filtered = filter_document(document, html=html, targets=targets)
    return [run['text'] for section in filtered['sections'] for block in section['blocks'] for run in block['runs']]


def filter_document(document, *, html, targets):
    """Keep selected DOM intervals, preserving source runs and exact code text.

    An unresolved fragment or degraded parser cannot safely define a boundary;
    in that case retain the raw audit capture and admit no body from that target.
    """
    document = copy.deepcopy(document)
    exclusions = [excluded for target in targets for excluded in target.get('excluded_targets', [])]
    whole_page = any(not target.get('fragment') and not target.get('heading_dom_path') for target in targets)
    if whole_page and not exclusions:
        return document
    if not targets:
        document.update(sections=[], parts=[], images={}, links={}, annotations={}, embedded_resources=[])
        return document
    if document['extraction']['mode'] != 'structured':
        document.update(sections=[], parts=[], images={}, links={}, annotations={}, embedded_resources=[])
        document['warnings'].append({'code':'selected_scope_requires_structured_boundaries'})
        return document
    soup = parse_html(html)
    indexer = Extractor(document['url'], document['source_id'])
    indexer.index_paths(soup)
    nodes = [soup, *soup.descendants]
    rank = {indexer.path(node):index for index, node in enumerate(nodes)}
    def target_range(target):
        fragment = target.get('fragment')
        boundary = soup.find(id=fragment) or soup.find('a', attrs={'name':fragment}) if fragment else None
        if boundary is None and target.get('heading_dom_path'):
            boundary = next((node for node in nodes if indexer.path(node) == target['heading_dom_path']), None)
        if not isinstance(boundary, Tag):
            return None
        if boundary.name not in HEADINGS:
            heading = boundary.find_parent(HEADINGS)
            if heading is None and boundary.name == 'a' and not boundary.get_text(strip=True):
                following = boundary.find_next(HEADINGS)
                # Empty named anchors immediately before a heading are common.
                if following is not None and not any(str(n).strip() for n in boundary.next_siblings
                                                      if isinstance(n, NavigableString)):
                    heading = following
            if heading is not None:
                boundary = heading
        start = rank.get(indexer.path(boundary))
        if start is None:
            return None
        if boundary.name in HEADINGS:
            following = next((node for node in boundary.find_all_next(HEADINGS)
                              if int(node.name[1]) <= int(boundary.name[1])), None)
            end = rank.get(indexer.path(following), len(nodes)) if following is not None else len(nodes)
        else:
            descendants = [boundary, *boundary.descendants]
            end = max(rank.get(indexer.path(node), start) for node in descendants) + 1
        return start, end

    ranges = [(0, len(nodes))] if whole_page else []
    if not whole_page:
        for target in targets:
            interval = target_range(target)
            if interval is None:
                document['warnings'].append({'code':'selected_scope_boundary_not_found', 'fragment':target.get('fragment')})
            else:
                ranges.append(interval)
    excluded_ranges = []
    for target in exclusions:
        interval = target_range(target)
        if interval is None:
            # A shared page cannot grant the entire unselected chapter when its
            # exclusion has become ambiguous or disappeared from the capture.
            ranges = []
            document['warnings'].append({'code':'selected_scope_shared_page_ambiguous', 'fragment':target.get('fragment')})
            break
        excluded_ranges.append(interval)

    def allowed(path):
        position = rank.get(path)
        return (position is not None and any(start <= position < end for start, end in ranges)
                and not any(start <= position < end for start, end in excluded_ranges))

    def trim(block):
        result = dict(block)
        retained_children = False
        for key in ('children', 'items', 'rows', 'cells'):
            if key in block:
                result[key] = [value for child in block[key] if (value := trim(child)) is not None]
                retained_children |= bool(result[key])
        for key in ('runs', 'caption_runs'):
            if key in block:
                result[key] = [run for run in block[key] if allowed(run.get('dom_path', block.get('dom_path')))]
        own = allowed(block.get('dom_path'))
        if not own and not retained_children and not result.get('runs'):
            return None
        return result

    sections = []
    for section in document['sections']:
        blocks = [value for block in section['blocks'] if (value := trim(block)) is not None]
        heading_selected = allowed(section.get('heading_dom_path'))
        if blocks or heading_selected:
            sections.append({**section, 'blocks':blocks,
                             'heading_runs':section['heading_runs'] if heading_selected else []})
    document['sections'] = sections
    blocks = [b for section in sections for b in section['blocks']]
    document['annotations'] = {key:value for key,value in document['annotations'].items() if allowed(value.get('dom_path'))}
    runs = [*all_runs(blocks), *(run for section in sections for run in inline_runs(section['heading_runs'])),
            *(run for annotation in document['annotations'].values() for run in inline_runs(annotation['runs']))]
    image_ids = {run['image_id'] for run in runs if run.get('image_id')}
    link_ids = {run['link_id'] for run in runs if run.get('link_id')}
    embedded_ids = {run['embedded_id'] for run in runs if run.get('embedded_id')}
    embedded_ids.update(block['embedded_id'] for block in all_blocks(blocks) if block.get('embedded_id'))
    document['images'] = {key:value for key,value in document['images'].items() if key in image_ids}
    document['links'] = {key:value for key,value in document['links'].items() if key in link_ids}
    document['embedded_resources'] = [value for value in document['embedded_resources'] if value.get('embedded_id') in embedded_ids]
    document['parts'] = []
    document['extraction']['selected_scope'] = True
    return document
