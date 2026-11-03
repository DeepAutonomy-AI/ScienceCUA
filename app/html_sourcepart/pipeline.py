"""Adapt the verified extractor to immutable native SourceParts and attachments."""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote_to_bytes, urljoin, urlparse

from bs4 import BeautifulSoup
import tinycss2

from ..html_navigation import html_base_url
from ..models import EvidenceImageRef, SourcePart
from . import fallback
from .extractor import Extractor, VERSION, IMPLEMENTATION_SHA256, all_blocks, plain


MAX_RESOURCE_BYTES = 4_000_000
MAX_STYLESHEETS = 24
MAX_IMAGES = 512


def source_json(store, value, *, kind, source_id, metadata=None):
    # These are captured document values, not runtime configuration. Preserve
    # attribute keys and text verbatim, including credential-shaped examples.
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8', errors='backslashreplace')
    return store.put_artifact(data, kind=kind, media_type='application/json', suffix='.json',
                              source_id=source_id, metadata=metadata)


def save_snapshot(workspace, snapshot, page_url, raw_artifact_id):
    value = dict(snapshot)
    value['raw_artifact_id'] = raw_artifact_id
    serialized = json.dumps(value, ensure_ascii=False).encode('utf-8', errors='backslashreplace')
    if len(serialized) > workspace.store.max_artifact_bytes:
        # A valid input can fit the raw artifact limit but exceed it after JSON
        # escaping. Keep the complete decoded snapshot in bounded text chunks.
        content = value.pop('content')
        chunks = []
        width = max(1, min(1_000_000, workspace.store.max_artifact_bytes // 12))
        for start in range(0, len(content), width):
            artifact = source_json(workspace.store, {'content':content[start:start+width]},
                                   kind='html_source_snapshot_chunk', source_id=workspace.manifest.source_id)
            workspace.allowed_artifacts.add(artifact.artifact_id)
            chunks.append({'artifact_id':artifact.artifact_id, 'start_offset':start, 'end_offset':min(len(content),start+width)})
        value['chunks'] = chunks
    return source_json(workspace.store, value, kind='html_source_snapshot', source_id=workspace.manifest.source_id,
                       metadata={'url':page_url, 'raw_artifact_id':raw_artifact_id})


class HtmlResources:
    """Bounded shared asset cache; resources never consume the child-page budget."""

    def __init__(self, workspace, page_url, resource_base):
        self.workspace = workspace
        self.local_roots = []
        self.local_hosts = {'', 'localhost'}
        for value in (resource_base, workspace.manifest.acquisition.get('local_resource_base_url'),
                      workspace.manifest.final_url, page_url):
            if value and urlparse(value).scheme == 'file':
                self.local_hosts.add(urlparse(value).netloc.casefold())
                path = workspace.acquisition._file_url_to_path(value).resolve()
                self.local_roots.append(path if str(value).endswith('/') else path.parent)
        if not hasattr(workspace, 'html_resource_cache'):
            workspace.html_resource_cache = {}
        self.cache = workspace.html_resource_cache

    def fetch(self, url, expected):
        self.workspace.acquisition._raise_if_cancelled()
        key = (url, expected)
        if key in self.cache:
            return self.cache[key]
        record = {'url': url, 'status': 'missing', 'resource_type': expected}
        try:
            scheme = urlparse(url).scheme.lower()
            if scheme == 'data' and expected == 'image':
                header, payload = url.split(',', 1)
                media = header[5:].split(';', 1)[0].lower()
                if len(payload) > MAX_RESOURCE_BYTES * 4:
                    raise ValueError('embedded image exceeds resource budget')
                data = base64.b64decode(payload, validate=True) if ';base64' in header else unquote_to_bytes(payload)
                final_url = url
            elif scheme in {'http', 'https', 'file'}:
                if scheme == 'file':
                    if urlparse(url).netloc.casefold() not in self.local_hosts:
                        raise ValueError('file resource host is outside the selected HTML source')
                    path = self.workspace.acquisition._file_url_to_path(url).resolve()
                    if not any(path == root or root in path.parents for root in self.local_roots):
                        raise ValueError('local resource is outside the selected HTML source directory')
                fetched = self.workspace.acquisition._read_url(url, max_bytes=MAX_RESOURCE_BYTES)
                data, media, final_url = fetched.data, fetched.media_type, fetched.final_url
            else:
                raise ValueError('unsupported or unresolved resource URL')
            if len(data) > MAX_RESOURCE_BYTES:
                raise ValueError('resource exceeds 4 MB budget')
            media = media.split(';', 1)[0].strip().lower()
            if expected == 'image' and media == 'application/octet-stream':
                if data.startswith(b'\x89PNG\r\n\x1a\n'):
                    media = 'image/png'
                elif data.startswith(b'\xff\xd8\xff'):
                    media = 'image/jpeg'
                elif data.startswith((b'GIF87a', b'GIF89a')):
                    media = 'image/gif'
                elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
                    media = 'image/webp'
            if expected == 'image' and not media.startswith('image/'):
                raise ValueError('resource did not return image content')
            if expected == 'stylesheet' and media not in {'text/css', 'text/plain', 'application/octet-stream'}:
                raise ValueError('resource did not return stylesheet content')
            suffix = mimetypes.guess_extension(media) or ('.css' if expected == 'stylesheet' else '.bin')
            artifact = self.workspace.store.put_artifact(
                data, kind='html_' + expected, media_type=media, suffix=suffix,
                source_id=self.workspace.manifest.source_id, metadata={'url': url, 'final_url': final_url},
            )
            record.update(status='saved', artifact_id=artifact.artifact_id, sha256=artifact.sha256,
                          byte_count=len(data), media_type=media, final_url=final_url)
            if expected == 'stylesheet':
                _rules, encoding = tinycss2.parse_stylesheet_bytes(data)
                record['text'] = encoding.codec_info.decode(data, errors='replace')[0]
        except Exception as exc:
            # Cancellation is an orchestration signal, never an asset failure.
            self.workspace.acquisition._raise_if_cancelled()
            record['reason'] = str(exc)[:400]
        self.cache[key] = record
        return record

    def resolve(self, soup, page_url, resource_base, reference):
        base = html_base_url(soup, page_url)
        if resource_base and not urlparse(reference).scheme and not reference.startswith('//'):
            local = urljoin(resource_base, reference)
            if urlparse(local).scheme == 'file':
                path = self.workspace.acquisition._file_url_to_path(local)
                if path.is_file() or urlparse(base).scheme not in {'http', 'https'}:
                    return local
        return urljoin(base, reference)

    def stylesheets(self, soup, page_url, resource_base):
        pending = [self.resolve(soup, page_url, resource_base, str(link['href'])) for link in soup.find_all('link', href=True)
                   if 'stylesheet' in link.get('rel', [])]
        seen, records = set(), []
        while pending and len(records) < MAX_STYLESHEETS:
            batch = list(dict.fromkeys(url for url in pending if url not in seen))[:MAX_STYLESHEETS-len(records)]
            if not batch:
                pending = []
                break
            seen.update(batch)
            with ThreadPoolExecutor(max_workers=6) as pool:
                fetched = list(pool.map(lambda url: self.fetch(url, 'stylesheet'), batch))
            records.extend(fetched)
            pending = []
            for record in fetched:
                if record['status'] != 'saved':
                    continue
                for rule in tinycss2.parse_stylesheet(record['text'], skip_comments=True, skip_whitespace=True):
                    if rule.type == 'at-rule' and rule.lower_at_keyword == 'import':
                        token = next((t for t in rule.prelude if t.type != 'whitespace'), None)
                        value = getattr(token, 'value', None)
                        if token and token.type == 'function' and token.lower_name == 'url':
                            value = tinycss2.serialize(token.arguments).strip().strip('"\'')
                        if value:
                            pending.append(urljoin(record['final_url'], value))
        if pending:
            records.append({'status': 'missing', 'resource_type': 'stylesheet', 'reason': 'stylesheet budget exceeded'})
        by_url = {r.get('url'):r for r in records if r['status'] == 'saved'}
        def expand(url, active):
            if url in active or url not in by_url:
                return ''
            record = by_url[url]
            result = []
            for rule in tinycss2.parse_stylesheet(record['text'], skip_comments=False, skip_whitespace=False):
                if rule.type == 'at-rule' and rule.lower_at_keyword == 'import':
                    token = next((t for t in rule.prelude if t.type != 'whitespace'), None)
                    value = getattr(token, 'value', None)
                    if token and token.type == 'function' and token.lower_name == 'url':
                        value = tinycss2.serialize(token.arguments).strip().strip('"\'')
                    if value:
                        tail = rule.prelude[rule.prelude.index(token)+1:]
                        if any(t.type not in {'whitespace','comment'} for t in tail):
                            # Preserve conditional imports without making their
                            # display rules unconditional in the static view.
                            result.append(tinycss2.serialize([rule]))
                        else:
                            result.append(expand(urljoin(record['final_url'], value), active | {url}))
                elif rule.type != 'error':
                    result.append(tinycss2.serialize([rule]))
            return '\n'.join(result)
        roots = [self.resolve(soup, page_url, resource_base, str(link['href'])) for link in soup.find_all('link', href=True)
                 if 'stylesheet' in link.get('rel', [])]
        if any(url not in seen for url in roots) and not any(r.get('reason')=='stylesheet budget exceeded' for r in records):
            records.append({'status':'missing', 'resource_type':'stylesheet', 'reason':'stylesheet budget exceeded'})
        return [expand(url, set()) for url in dict.fromkeys(roots) if url in by_url], [{k:v for k,v in r.items() if k != 'text'} for r in records]

    def images(self, document):
        urls = list(dict.fromkeys(i['url'] for i in document['images'].values()))
        admitted = urls[:MAX_IMAGES]
        with ThreadPoolExecutor(max_workers=6) as pool:
            fetched = dict(zip(admitted, pool.map(lambda url: self.fetch(url, 'image'), admitted)))
        result = {}
        for image_id, image in document['images'].items():
            resource = fetched.get(image['url'], {'status':'missing', 'reason':'image resource budget exceeded'})
            result[image_id] = {**image, **resource}
        return result


def pipeline_sections(document):
    """Keep H1-H4 topology, inline H5/H6, and authored disclosure groups."""
    output = []
    for section in document['sections']:
        section = copy.copy(section)
        section['blocks'] = list(section['blocks'])
        if section['level'] > 4 and output:
            heading = {'type':'heading', 'level':section['level'], 'runs':section['heading_runs'],
                       'dom_path':section.get('heading_dom_path', '')}
            output[-1]['blocks'].extend([heading, *section['blocks']])
            continue
        pending = []
        heading_emitted = False
        for block in section['blocks']:
            if block['type'] != 'disclosure':
                pending.append(block)
                continue
            if pending or (section['heading_runs'] and not heading_emitted):
                output.append({**section, 'blocks':pending,
                               'heading_runs':[] if heading_emitted else section['heading_runs']})
                heading_emitted = True
                pending = []
            label = plain(block['runs'], image_marker=False).strip() or section['title']
            output.append({**section, 'title':label, 'level':min(4, max(1, section['level']+1)),
                           'heading_path':[*section['heading_path'], label], 'heading_runs':[],
                           'blocks':[block], 'navigation_group_label':label,
                           'navigation_group_dom_id':block.get('anchor_id')})
        if pending or not heading_emitted and section['heading_runs']:
            output.append({**section, 'blocks':pending,
                           'heading_runs':[] if heading_emitted else section['heading_runs']})
    return output


def semantic_nodes(document):
    """Bounded compatibility index; the per-part AST remains the full structure."""
    result, headings = [], []
    for section in document['sections']:
        level = int(section['level'])
        if level:
            while headings and headings[-1][0] >= level:
                headings.pop()
            result.append({'ordinal':len(result)+1, 'tag':'h'+str(level), 'text':section['title'][:10_000],
                           'heading_level':level, 'heading_path':section['heading_path'],
                           'parent_heading_ordinal':headings[-1][1] if headings else None,
                           'dom_path':section.get('heading_dom_path')})
            headings.append((level,len(result)))
        for block in all_blocks(section['blocks']):
            if len(result) >= 10_000:
                return result
            tag = {'paragraph':'p','code':'pre','table':'table','list_item':'li','caption':'figcaption'}.get(block['type'])
            if tag:
                result.append({'ordinal':len(result)+1, 'tag':tag, 'text':plain(block.get('runs', []))[:10_000],
                               'heading_path':section['heading_path'], 'dom_path':block['dom_path']})
    return result


def extract_page(workspace, **arguments):
    """Keep recovery available across adaptation and artifact-size failures too."""
    try:
        return _extract_page(workspace, **arguments)
    except (OSError, MemoryError):
        # Recovery cannot repair a storage failure or an exhausted process.
        raise
    except Exception as exc:
        workspace.acquisition._raise_if_cancelled()
        html = arguments['html']
        url = arguments['page_url']
        probe, _ = fallback.scan(html, url, fallback.Limits(), arguments.get('resource_base_url'), arguments.get('resource_urls'))
        recovered = fallback.recover(html, url, workspace.manifest.source_id, 10_000, VERSION, IMPLEMENTATION_SHA256,
                                     [{'code':'pipeline_structure_adaptation_failed', 'error_type':type(exc).__name__, 'message':str(exc)[:400]}], probe)
        retry = {**arguments, 'stylesheets':[], '_recovered_document':recovered}
        return _extract_page(workspace, **retry)


def _extract_page(workspace, *, html, soup, page_url, raw_artifact_id, page_part_id, page_ordinal,
                  stylesheets=None, resource_base_url=None, resource_urls=None, fetch_resources=True,
                  scope_targets=None, _recovered_document=None):
    """Return the core document, native parts, and persisted page-level metadata."""
    resources = HtmlResources(workspace, page_url, resource_base_url)
    stylesheet_records = []
    if stylesheets is None:
        stylesheets, stylesheet_records = resources.stylesheets(soup, page_url, resource_base_url) if fetch_resources else ([], [])
    if resource_urls is None:
        resource_urls = {str(image['src']):resources.resolve(soup, page_url, resource_base_url, str(image['src']))
                         for image in soup.find_all('img', src=True)}
    extractor_type = Extractor
    if workspace.manifest.source_metadata.get('html_processing_mode') == 'model_guided':
        from ..html_guided.extractor import GuidedExtractor
        extractor_type = GuidedExtractor
    extractor = extractor_type(page_url, workspace.manifest.source_id, resource_base_url, stylesheets, resource_urls)
    document = _recovered_document if _recovered_document is not None else extractor.extract(html)
    snapshot = save_snapshot(workspace, document['source_snapshot'], page_url, raw_artifact_id)
    if scope_targets is not None:
        from .scope import filter_document
        document = filter_document(document, html=html, targets=scope_targets)
        extractor.annotations = document.get('annotations', {})
        for warning in document.get('warnings', []):
            if str(warning.get('code', '')).startswith('selected_scope_'):
                workspace.gaps.append(f'HTML page {page_url}: {warning["code"]}; unresolvable selected content was excluded.')
    images = resources.images(document) if fetch_resources else copy.deepcopy(document['images'])
    if document['extraction']['mode'] == 'structured':
        adapted = {**document, 'sections':pipeline_sections(document)}
        raw_parts = extractor.make_parts(adapted, 10_000)
    else:
        adapted, raw_parts = document, document['parts']
    for raw in raw_parts:
        if not raw['content'] and raw['metadata']['image_ids']:
            # A heading can contain only an image. Give the existing image a
            # transport anchor so native text-span Evidence can carry it too.
            raw['content'] = '\n'.join('[image ' + key + ']' for key in raw['metadata']['image_ids'])
            raw['metadata']['synthetic_image_anchor'] = True
        raw['metadata'].update(extraction_mode=document['extraction']['mode'],
                               degraded=document['extraction']['degraded'], source_snapshot_sha256=document['raw_sha256'])
    headings, section_primaries, parts, structure_ids = [], {}, [], []
    for raw in raw_parts:
        locator = dict(raw['locator'])
        index, chunk = locator['section_index'], locator['chunk']
        level = int(locator.get('heading_level') or 0)
        section = adapted['sections'][index-1] if document['extraction']['mode'] == 'structured' else {}
        if chunk == 1:
            while headings and headings[-1][0] >= level:
                headings.pop()
        parent = section_primaries.get(index) if chunk > 1 else (headings[-1][1] if headings else page_part_id)
        section_primaries.setdefault(index, raw['part_id'])
        if chunk == 1 and level > 0:
            headings.append((level, raw['part_id']))
        blocks = raw['metadata']['blocks']
        block_types = {b['type'] for b in all_blocks(blocks)}
        structure = source_json(workspace.store, {
            'schema_version':VERSION, 'source_id':workspace.manifest.source_id, 'source_part_id':raw['part_id'],
            'blocks':blocks, 'block_ranges':raw['metadata']['block_ranges'],
            'heading_runs':raw['metadata'].get('heading_runs', []), 'annotations':raw['metadata'].get('annotations', []),
            'source_snapshot_artifact_id':snapshot.artifact_id, 'extraction':document['extraction'],
        }, kind='html_sourcepart_structure', source_id=workspace.manifest.source_id, metadata={'url':page_url})
        structure_ids.append(structure.artifact_id)
        links = {key:document['links'][key] for key in raw['metadata']['link_ids']}
        part_images = {key:images[key] for key in raw['metadata']['image_ids']}
        embedded = [r for r in document['embedded_resources'] if r.get('embedded_id') in raw['metadata'].get('embedded_ids', [])]
        metadata = {k:v for k,v in raw['metadata'].items() if k not in {'blocks','block_ranges','annotations','heading_runs'}}
        metadata.update(structure_artifact_id=structure.artifact_id, source_snapshot_artifact_id=snapshot.artifact_id,
                        links=links, images=part_images, embedded_resources=embedded,
                        content_kinds=sorted(block_types), container=not bool(raw['content']),
                        evidence_default=bool(raw['content']), semantic_unit=True)
        if section.get('navigation_group_label'):
            metadata.update(semantic_container='details', navigation_group_label=section['navigation_group_label'],
                            navigation_group_dom_id=section.get('navigation_group_dom_id'))
        image_refs = []
        for image_id, image in part_images.items():
            if image.get('status') != 'saved':
                continue
            anchor = raw['content'].find('[image ' + image_id + ']')
            image_refs.append(EvidenceImageRef(
                artifact_id=image['artifact_id'], source_part_id=raw['part_id'], media_type=image['media_type'],
                page=page_ordinal, role='unknown', association_method=(
                    'html_dom_sourcepart' if document['extraction']['mode']=='structured'
                    else 'html_basic_text_marker' if document['extraction']['mode']=='basic_text'
                    else 'html_document_fallback_sidecar'),
                anchor_offset=max(0, min(anchor, len(raw['content'])-1)),
            ).model_dump(mode='json'))
        metadata['image_refs'] = image_refs
        metadata['multimodal'] = bool(image_refs and raw['content'])
        locator.update(section_path=' > '.join(locator.get('heading_path') or [locator.get('heading', raw['label'])]),
                       structure_artifact_id=structure.artifact_id, source_snapshot_artifact_id=snapshot.artifact_id,
                       extraction_mode=document['extraction']['mode'])
        if links:
            locator['link_targets_artifact_id'] = None  # Filled after the page sidecar is persisted.
        if parent != page_part_id:
            locator['parent_section_part_id'] = parent
        kind = raw['kind']
        if section.get('navigation_group_label'):
            kind = 'html_navigation_group'
        elif raw['content'] and block_types <= {'code','code_group'}:
            kind = 'html_code'
        part = SourcePart(part_id=raw['part_id'], source_id=workspace.manifest.source_id, kind=kind, label=raw['label'][:300],
                          locator=locator, parent_part_id=parent, ordinal=raw['ordinal'], artifact_id=raw_artifact_id,
                          char_count=len(raw['content']), metadata=metadata)
        parts.append((part, raw['content']))
    summary = {k:v for k,v in document.items() if k not in {'source_snapshot','parts','sections'}}
    summary.update(source_snapshot_artifact_id=snapshot.artifact_id, raw_artifact_id=raw_artifact_id,
                   images=images, stylesheets=stylesheet_records,
                   sections=[{k:v for k,v in s.items() if k not in {'blocks','heading_runs'}} for s in adapted['sections']],
                   parts=[{'part_id':p.part_id, 'structure_artifact_id':p.metadata['structure_artifact_id']} for p,_ in parts])
    artifact = source_json(workspace.store, summary, kind='html_structured_source', source_id=workspace.manifest.source_id,
                           metadata={'url':page_url, 'extractor_version':VERSION})
    for part, _ in parts:
        if 'link_targets_artifact_id' in part.locator:
            part.locator['link_targets_artifact_id'] = artifact.artifact_id
    permitted = [snapshot.artifact_id, artifact.artifact_id, *structure_ids,
                 *(i['artifact_id'] for i in images.values() if i.get('artifact_id')),
                 *(s['artifact_id'] for s in stylesheet_records if s.get('artifact_id'))]
    workspace.allowed_artifacts.update(permitted)
    missing = sum(i.get('status') != 'saved' for i in images.values()) if fetch_resources else 0
    if missing:
        workspace.warnings.append(f'HTML page {page_url}: {missing} image resources could not be saved; their source references remain in {artifact.artifact_id}.')
    if any(s['status'] != 'saved' for s in stylesheet_records):
        workspace.warnings.append(f'HTML page {page_url}: some author stylesheets were unavailable; saved structure records the stylesheet failures.')
    if document['extraction']['degraded']:
        reasons = ', '.join(r['code'] for r in document['extraction']['reasons'])
        workspace.gaps.append(f'HTML page {page_url} used {document["extraction"]["mode"]} recovery: {reasons}. Complete captured HTML is retained in {snapshot.artifact_id}.')
    info = {'extractor_version':VERSION, 'implementation_sha256':IMPLEMENTATION_SHA256,
            'structured_source_artifact_id':artifact.artifact_id, 'source_snapshot_artifact_id':snapshot.artifact_id,
            'extraction':document['extraction'], 'image_count':len(images), 'saved_image_count':sum(i.get('status')=='saved' for i in images.values()),
            'missing_image_count':missing, 'embedded_count':len(document['embedded_resources'])}
    workspace.store.emit('html_sourceparts_extracted', {'url':page_url, 'part_count':len(parts), **info},
                         source_id=workspace.manifest.source_id)
    return document, parts, info
