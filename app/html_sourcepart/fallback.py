"""Independent, standard-library recovery for already captured HTML.

No DOM, CSS, software-specific selectors, downloads or script execution. Every
result retains the complete input snapshot; raw recovery parts concatenate to
that input exactly. Synthetic notices are explicitly marked, never page prose.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse


def sha(value):
    return hashlib.sha256(value.encode('utf-8', errors='surrogatepass')).hexdigest()


@dataclass(frozen=True)
class Limits:
    structured_chars: int = 8_000_000
    scan_chars: int = 16_000_000
    nodes: int = 100_000
    depth: int = 192
    span: int = 1024
    grid_slots: int = 250_000
    resources: int = 10_000
    fallback_parts: int = 10_000

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + ' must be a positive integer')


class ScanBudgetExceeded(Exception):
    def __init__(self, code, limit):
        self.code, self.limit = code, limit
        super().__init__(code)


VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
BREAKS = {'p', 'div', 'section', 'article', 'main', 'header', 'footer', 'nav', 'aside', 'li', 'dt', 'dd',
          'ul', 'ol', 'dl', 'tr', 'table', 'blockquote', 'pre', 'figure', 'figcaption', 'details',
          'summary', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr'}


def resolve(base, reference):
    try:
        return urljoin(base, reference)
    except (ValueError, TypeError):
        return reference


def clickable(url):
    try:
        return urlparse(url).scheme.lower() in {'http', 'https', 'file', 'mailto'}
    except ValueError:
        return False


class Probe(HTMLParser):
    """Iterative conservative text/resource capture and complexity preflight."""
    def __init__(self, url, limits):
        super().__init__(convert_charrefs=True)
        self.url, self.base, self.limits = url, url, limits
        self.stack, self.pieces, self.refs, self.reasons = [], [], [], []
        self.links, self.images, self.embedded = {}, {}, []
        self.anchor_ids, self.href_ids = [], {}
        self.title_pieces = []
        self.skip_tag = None
        self.in_head = self.in_title = 0
        self.depth_tracking_stopped = False
        self.canvas_seen = 0
        self.title_chars = 0
        self.length = self.nodes = self.maximum_depth = self.grid_slots = self.content_chars = self.scripts = 0

    def issue(self, code, **details):
        if not any(r['code'] == code for r in self.reasons):
            self.reasons.append({'code': code, **details})

    def append(self, value, kind=None, ref_id=None):
        start = self.length
        self.pieces.append(value)
        self.length += len(value)
        if kind:
            self.refs.append({'kind': kind, 'id': ref_id, 'start': start, 'end': self.length})

    def suppressed(self):
        return self.skip_tag is not None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.nodes += 1
        if self.nodes > self.limits.nodes:
            raise ScanBudgetExceeded('scan_node_budget_exceeded', self.limits.nodes)
        # Approximate common implied closes; malformed old paragraphs should
        # not inflate the preflight depth indefinitely.
        if not self.depth_tracking_stopped and tag in BREAKS and 'p' in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index('p')]
        groups = [{'li'}, {'dt', 'dd'}, {'td', 'th'}, {'tr'}, {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}]
        for group in groups:
            if not self.depth_tracking_stopped and tag in group and self.stack and self.stack[-1] in group:
                self.stack.pop()
        if tag not in VOID and not self.depth_tracking_stopped:
            self.stack.append(tag)
        self.maximum_depth = max(self.maximum_depth, len(self.stack))
        if self.maximum_depth > self.limits.depth:
            self.issue('structured_depth_budget_exceeded', limit=self.limits.depth)
            self.depth_tracking_stopped = True
        self.in_head += int(tag == 'head')
        self.in_title += int(tag == 'title')
        if tag in {'script', 'style'}:
            self.skip_tag = tag
        if tag == 'script':
            self.scripts += 1
        if self.suppressed():
            return
        if tag == 'canvas':
            self.canvas_seen += 1
        if tag == 'base' and attributes.get('href'):
            self.base = resolve(self.url, attributes['href'])
        if tag in {'td', 'th'}:
            spans = []
            for name in ('rowspan', 'colspan'):
                value = attributes.get(name, '1') or '1'
                try:
                    if not re.fullmatch(r'\+?[0-9]+', value.strip()):
                        raise ValueError('not an HTML nonnegative integer')
                    number = int(value)
                    if number <= 0:
                        self.issue('table_span_not_supported', attribute=name, value=value[:80])
                    elif number > self.limits.span:
                        self.issue('table_span_budget_exceeded', attribute=name, limit=self.limits.span)
                    spans.append(max(1, min(number, self.limits.span + 1)))
                except ValueError:
                    self.issue('table_span_not_integer', attribute=name, value=value[:80])
                    spans.append(1)
            self.grid_slots += spans[0] * spans[1]
            if self.grid_slots > self.limits.grid_slots:
                self.issue('table_grid_budget_exceeded', limit=self.limits.grid_slots)
        if tag in BREAKS:
            self.append('\n')
        elif tag in {'td', 'th'}:
            self.append('\t')
        path = 'capture/' + tag + '@' + ':'.join(map(str, self.getpos()))
        if tag in {'a', 'img', 'frame', 'iframe', 'embed', 'object', 'source', 'video', 'audio'} and (
                len(self.images) + len(self.links) + len(self.embedded) >= self.limits.resources):
            raise ScanBudgetExceeded('scan_resource_budget_exceeded', self.limits.resources)
        if tag == 'a' and attributes.get('href') is not None:
            href = attributes['href'] or ''
            key = self.href_ids.setdefault(href, 'L' + str(len(self.href_ids) + 1))
            self.links.setdefault(key, {'link_id': key, 'href': href, 'occurrences': [], 'labels': []})['occurrences'].append(path)
            self.anchor_ids.append(key)
        if tag == 'img':
            key = 'I' + str(len(self.images) + 1)
            src = attributes.get('src') or attributes.get('data-src') or ''
            self.images[key] = {'image_id': key, 'src': src, 'alt': attributes.get('alt') or '',
                                'title': attributes.get('title') or '', 'dom_path': path, 'role': 'unknown',
                                'status': 'pending' if src else 'missing_src', 'display_context': 'standalone',
                                'author_classes': (attributes.get('class') or '').split(),
                                'author_width': attributes.get('width'), 'author_height': attributes.get('height'),
                                'author_style': attributes.get('style') or '', 'author_layout': {}}
            label = '[image ' + key + (': ' + self.images[key]['alt'] if self.images[key]['alt'] else '') + ']'
            self.append('\n' + label + '\n', 'image', key)
        for attribute in ('src', 'srcset', 'data', 'poster'):
            if attributes.get(attribute) and (tag in {'frame', 'iframe', 'embed', 'object', 'video', 'audio', 'source'} or attribute == 'srcset'):
                self.embedded.append({'tag': tag, 'attribute': attribute, 'reference': attributes[attribute], 'dom_path': path})

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        suppressed = self.suppressed()
        if self.skip_tag == tag:
            self.skip_tag = None
        self.in_head = max(0, self.in_head - int(tag == 'head'))
        self.in_title = max(0, self.in_title - int(tag == 'title'))
        if not self.depth_tracking_stopped and tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]
        if suppressed:
            return
        if tag == 'a' and self.anchor_ids:
            key = self.anchor_ids.pop()
            self.append(' [' + key + ']', 'link', key)
        if tag in BREAKS:
            self.append('\n')

    def handle_data(self, value):
        if self.suppressed():
            return
        if self.in_title:
            remaining = 1025 - self.title_chars
            if remaining > 0:
                self.title_pieces.append(value[:remaining])
                self.title_chars += len(value[:remaining])
            if len(value) > remaining:
                self.issue('structured_title_budget_exceeded', limit=1024)
        if not self.in_head:
            self.content_chars += len(value.strip())
        self.append(value)
        if self.anchor_ids and value.strip():
            self.links[self.anchor_ids[-1]]['labels'].append(value.strip())

    def finish(self, resource_base=None, resource_urls=None):
        for entry in self.links.values():
            entry['url'] = resolve(self.base, entry['href'])
            entry['clickable'] = clickable(entry['url'])
            entry['kind'] = 'web' if entry['clickable'] else 'non_web'
            self.append('\n[' + entry['link_id'] + ']: ' + entry['url'], 'link', entry['link_id'])
        for entry in self.images.values():
            entry['url'] = (resource_urls or {}).get(entry['src'], resolve(resource_base or self.base, entry['src'])) if entry['src'] else ''
        for entry in self.embedded:
            entry['url'] = resolve(resource_base or self.base, entry['reference']) if entry['attribute'] != 'srcset' else None
        return self


def scan(value, url, limits, resource_base=None, resource_urls=None):
    if len(value) > limits.scan_chars:
        return None, [{'code': 'scan_input_budget_exceeded', 'limit': limits.scan_chars}]
    try:
        probe = Probe(url, limits)
        probe.feed(value)
        probe.close()
        probe.finish(resource_base, resource_urls)
        if len(value) > limits.structured_chars:
            probe.issue('structured_input_budget_exceeded', limit=limits.structured_chars)
        return probe, probe.reasons
    except ScanBudgetExceeded as exc:
        return None, [{'code': exc.code, 'limit': exc.limit}]
    except Exception as exc:
        return None, [{'code': 'basic_parser_failed', 'error_type': type(exc).__name__, 'message': str(exc)[:400]}]


def source_snapshot(value):
    return {'format': 'html', 'content': value, 'char_count': len(value), 'sha256': sha(value), 'complete': True}


def recover(value, url, source_id, max_chars, version, implementation_sha256, reasons, probe=None, max_parts=10_000):
    """Do not call any structured formatter, parser or SourcePart builder."""
    useful = probe is not None and (probe.content_chars > 0 or any(i['src'] for i in probe.images.values()))
    mode = 'basic_text' if useful else 'raw_html' if value else 'empty_capture'
    content = ''.join(probe.pieces) if useful else value if value else '[The captured HTML is empty; no source content was available.]'
    title = re.sub(r'\s+', ' ', ''.join(probe.title_pieces)).strip() if probe else ''
    title = (title or url or 'Captured HTML')[:256]
    snapshot = source_snapshot(value)
    parts, blocks = [], []
    chunk_size = max(max_chars, (len(content) + max_parts - 1) // max_parts)
    if chunk_size > max_chars:
        reasons = [*reasons, {'code': 'fallback_part_count_budget', 'limit': max_parts, 'chunk_size': chunk_size}]
    count = max(1, (len(content) + chunk_size - 1) // chunk_size)
    references = probe.refs if mode == 'basic_text' else []
    reference_index = 0
    for index in range(count):
        start, end = index * chunk_size, min(len(content), (index + 1) * chunk_size)
        text = content[start:end]
        path = 'fallback/chunk[' + str(index + 1) + ']'
        kind = 'fallback_html' if mode == 'raw_html' else 'fallback_text'
        block = {'type': kind, 'dom_path': path,
                 'runs': [{'type': 'text', 'text': text, 'styles': [], 'dom_path': path + '/text()[1]'}]}
        blocks.append(block)
        # References are emitted in increasing, nonoverlapping text ranges.
        # Do not rescan the whole catalog for every small chunk.
        while reference_index < len(references) and references[reference_index]['end'] <= start:
            reference_index += 1
        refs, next_index = [], reference_index
        while next_index < len(references) and references[next_index]['start'] < end:
            refs.append(references[next_index])
            next_index += 1
        image_ids = sorted({r['id'] for r in refs if r['kind'] == 'image'})
        link_ids = sorted({r['id'] for r in refs if r['kind'] == 'link'})
        # Raw fallback uses document-scope sidecars; no synthetic DOM ownership
        # or image position is invented for its raw chunks.
        if index == 0 and probe and mode != 'basic_text':
            image_ids, link_ids = list(probe.images), list(probe.links)
        label = title + (' (' + str(index + 1) + ')' if count > 1 else '')
        parts.append({'part_id': 'part_' + sha(source_id + '|' + url + '|' + mode + '|' + str(index) + '|' + version)[:20],
                      'source_id': source_id, 'kind': 'html_' + mode, 'label': label,
                      'locator': {'url': url, 'section_index': 1, 'heading': title, 'heading_level': 0,
                                  'heading_path': [], 'chunk': index + 1, 'capture_start': start, 'capture_end': end},
                      'ordinal': 10_000 + index + 1, 'content': text, 'content_sha256': sha(text), 'char_count': len(text),
                      'metadata': {'normalization_version': version, 'format': 'html' if mode == 'raw_html' else 'text',
                                   'blocks': [block], 'block_ranges': [{'dom_path': path, 'block_type': kind, 'start_offset': 0, 'end_offset': len(text)}],
                                   'heading_runs': [], 'annotations': [], 'structural_heading': False,
                                   'image_ids': image_ids, 'link_ids': link_ids, 'soft_limit_exceeded': len(text) > max_chars,
                                   'extraction_mode': mode, 'degraded': True, 'synthetic_notice': mode == 'empty_capture'}})
        parts[-1]['metadata']['source_snapshot_sha256'] = snapshot['sha256']
    return {'schema_version': version, 'implementation_sha256': implementation_sha256,
            'source_id': source_id, 'url': url, 'title': title, 'parser': 'stdlib.HTMLParser' if useful else 'raw_snapshot',
            'raw_sha256': snapshot['sha256'], 'source_snapshot': snapshot,
            'content_root': {'tag': 'capture', 'role': None, 'dom_path': 'capture', 'dom_path_kind': 'synthetic'},
            'sections': [{'title': title, 'level': 0, 'heading_path': [], 'heading_runs': [], 'blocks': blocks}],
            'parts': parts, 'links': probe.links if probe else {}, 'images': probe.images if probe else {},
            'embedded_resources': probe.embedded if probe else [], 'annotations': {},
            'style_context': {'external_stylesheets': 0, 'mode': 'fallback_css_not_applied'},
            'extraction': {'mode': mode, 'degraded': True, 'reasons': reasons,
                           'capture_status': 'empty' if not value else 'saved',
                           'static_text_available': bool(probe.content_chars) if probe else None,
                           'render_required': bool(not useful and (probe.scripts or probe.embedded or probe.canvas_seen)) if probe else None,
                           'embedded_sources_pending': bool(probe and probe.embedded),
                           'coverage': 'entire_capture_excluding_script_style' if useful else 'complete_raw_html'},
            'warnings': [*reasons, {'code': 'extraction_degraded', 'mode': mode}]}


def validate_recovery(document, value):
    snapshot = document.get('source_snapshot', {})
    parts = document.get('parts', [])
    mode = document.get('extraction', {}).get('mode')
    combined = ''.join(p['content'] for p in parts)
    offsets_ok, cursor = True, 0
    for part in parts:
        offsets_ok = offsets_ok and part['locator']['capture_start'] == cursor
        cursor += part['char_count']
        offsets_ok = offsets_ok and part['locator']['capture_end'] == cursor
    return {'snapshot_complete': snapshot.get('content') == value and snapshot.get('sha256') == sha(value)
                                 and snapshot.get('char_count') == len(value) and snapshot.get('complete') is True,
            'parts_exist': bool(parts),
            'part_ids_unique': len({p['part_id'] for p in parts}) == len(parts),
            'part_hashes_match': all(p['content_sha256'] == sha(p['content']) and p['char_count'] == len(p['content']) for p in parts),
            'chunk_offsets_match': offsets_ok,
            'block_text_matches': all(''.join(r['text'] for b in p['metadata']['blocks'] for r in b['runs']) == p['content'] for p in parts),
            'references_exist': all(set(p['metadata']['image_ids']) <= set(document['images'])
                                    and set(p['metadata']['link_ids']) <= set(document['links']) for p in parts),
            'raw_chunks_complete': combined == value if mode == 'raw_html' else None,
            'degraded': True, 'mode': mode}
