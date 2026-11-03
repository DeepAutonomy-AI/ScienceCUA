"""Generic PDF -> structured SourceParts.

Geometry and typography are evidence, never software-specific text keywords.
Every original line is assigned to a block or retained as a page annotation.
The original PDF and page snapshots remain authoritative when inference fails.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    fitz = None
from . import fallback
from .filesystem import filesystem_path

VERSION = 'pdf-sourcepart.v4-visual-recovery'


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def render_png(page, **kwargs):
    pixmap = fallback.decode_call(page.get_pixmap, **kwargs)
    return fallback.decode_call(pixmap.tobytes, 'png')


def file_digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def box_union(boxes):
    boxes = [b for b in boxes if b]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def text_key(value):
    return re.sub(r'\W+', '', value.casefold())


def plain(runs):
    return ''.join(r.get('text', '') for r in runs)


def is_mono(run):
    return bool(run.get('flags', 0) & 8) or bool(re.search(r'mono|courier|consolas|typewriter', run.get('font', ''), re.I))


def safe_link(value):
    return bool(re.match(r'^(https?://|mailto:|#)', value or '', re.I))


def inline(runs):
    output = []
    # Coalesce spans with the same style: PDF producers often write one letter
    # per span. Formatting every letter destroys readability and token cost.
    groups = []
    for run in runs:
        style = (run.get('flags', 0) & 19, is_mono(run), run.get('link_id'))
        if groups and groups[-1][0] == style:
            groups[-1][1] += run.get('text', '')
        else:
            groups.append([style, run.get('text', '')])
    for (flags, mono, link), text in groups:
        left = text[:len(text)-len(text.lstrip())]
        right = text[len(text.rstrip()):]
        middle = text.strip()
        if not middle:
            output.append(text); continue
        escaped = re.sub(r'([\\`*\[\]])', r'\\\1', middle)
        if mono:
            fence = '`' * (max([len(x) for x in re.findall(r'`+', middle)] or [0]) + 1)
            escaped = fence + middle + fence
        elif flags & 16:
            escaped = '**' + escaped + '**'
        elif flags & 2:
            escaped = '*' + escaped + '*'
        if flags & 1:
            escaped = '<sup>' + escaped + '</sup>'
        if link:
            escaped = '[' + escaped + '][' + link + ']'
        output.append(left + escaped + right)
    return ''.join(output)


def block_markdown(block):
    kind = block['type']
    if kind == 'heading':
        return '#' * min(6, block['level']) + ' ' + inline(block['runs']).strip()
    if kind == 'code':
        body = block['text'].rstrip('\n')
        fence = '`' * max(3, 1+max([len(x) for x in re.findall(r'`+', body)] or [0]))
        return fence + '\n' + body + '\n' + fence
    if kind == 'table':
        rows = [[inline(cell.get('runs', [])).strip().replace('|', '\\|').replace('\n', '<br>')
                 for cell in row['cells']] for row in block['rows']]
        width = max([len(row) for row in rows] or [1])
        rows = [row + ['']*(width-len(row)) for row in rows]
        if not rows:
            return ''
        rendered = ['| ' + ' | '.join(row) + ' |' for row in rows]
        # Do not invent a header for an unlabelled keyboard/parameter table.
        if block.get('header'):
            rendered.insert(1, '| ' + ' | '.join(['---']*width) + ' |')
        return '\n'.join(rendered)
    if kind == 'list':
        return '\n'.join(('  '*item.get('depth', 0)) + item.get('marker', '-') + ' ' +
                         inline(item['runs']).strip() for item in block['items'])
    if kind == 'toc':
        return '\n'.join('- [' + entry['title'] + '](#page-' + str(entry['target_page']) + ') — ' +
                         str(entry['printed_page']) for entry in block['entries'])
    if kind == 'image':
        return '![Image ' + block['image_id'] + '](assets/' + block['filename'] + ')'
    if kind == 'page_image':
        text = block.get('text','[Page visual retained.]')
        return text + ('\n\n![Original page image](assets/' + block['filename'] + ')' if block.get('filename') else '')
    return inline(block.get('runs', [])).strip() if block.get('runs') else block.get('text', '')


def line_runs(line):
    return line['runs']


def join_lines(lines, *, preserve_breaks=False):
    runs = []
    for i, line in enumerate(lines):
        if i:
            previous = lines[i-1]['text'].rstrip()
            current = line['text'].lstrip()
            # Only discretionary soft hyphens are unambiguously removable.
            # A visible con-/verted stays source-faithful; the page image and
            # exact span snapshot allow later language-aware repair.
            sep = '\n' if preserve_breaks else ' '
            if previous.endswith('\u00ad') and current[:1].islower() and runs:
                runs[-1] = {**runs[-1], 'text':runs[-1]['text'].rstrip('\u00ad')}
                sep = ''
            elif not preserve_breaks and re.search(r'[A-Za-z]-$',previous) and current[:1].islower():
                # Retain the literal hyphen, but do not introduce a spurious
                # space into a wrapped compound word. Removing visible hyphens
                # requires linguistic evidence beyond this layout experiment.
                sep = ''
            runs.append({'text':sep, 'synthetic':'line_join'})
        runs.extend(line_runs(line))
    return runs


def make_block(kind, lines, **extra):
    runs = join_lines(lines, preserve_breaks=kind == 'code')
    return {'type':kind, 'page':lines[0]['page'] if lines else extra.pop('page', None),
            'bbox':box_union([line['bbox'] for line in lines]),
            'line_ids':[line['line_id'] for line in lines], 'runs':runs,
            'text':plain(runs), **extra}


def raw_page(page, page_number):
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    layout = page.get_text('dict', flags=flags)
    links = []
    for n, record in enumerate(page.get_links(), 1):
        url = record.get('uri')
        if not url and record.get('kind') == fitz.LINK_GOTO:
            url = '#page-' + str(record.get('page', 0)+1)
        links.append({'link_id':f'L{page_number}_{n}', 'url':url,
                      'target_page':record.get('page'), 'bbox':list(record['from']),
                      'clickable':safe_link(url), 'kind':record.get('kind')})
    lines = []
    for block_index, block in enumerate(layout['blocks']):
        if block['type'] != 0:
            continue
        for line_index, line in enumerate(block['lines']):
            spans = []
            for span in line['spans']:
                run = {key:span.get(key) for key in ('text','font','size','flags','color','origin','bbox')}
                for link in links:
                    if intersection(run['bbox'], link['bbox']) > area(run['bbox'])*.25:
                        run['link_id'] = link['link_id']; break
                spans.append(run)
            text = plain(spans)
            if not text.strip():
                continue
            weight = sum(len(s['text']) for s in spans) or 1
            lines.append({'line_id':f'P{page_number}.B{block_index}.L{line_index}', 'page':page_number,
                          'block_index':block_index, 'bbox':list(line['bbox']), 'direction':list(line['dir']),
                          'runs':spans, 'text':text, 'size':max(s['size'] for s in spans),
                          'bold_ratio':sum(len(s['text']) for s in spans if s['flags'] & 16)/weight,
                          'mono_ratio':sum(len(s['text']) for s in spans if is_mono(s))/weight})
    return {'page':page_number, 'width':float(page.rect.width), 'height':float(page.rect.height),
            'raw_text':page.get_text('text', sort=True), 'lines':lines, 'links':links, 'rotation':page.rotation}


def furniture_keys(pages):
    seen = defaultdict(set)
    for page in pages:
        for line in page.get('lines', []):
            y = line['bbox'][1]/page['height']
            if y < .125 or y > .91:
                key = furniture_key(line['text'])
                seen[key].add(page['page'])
    minimum = max(3, math.ceil(len(pages)*.25))
    return {key for key, occurrences in seen.items() if len(occurrences) >= minimum}


def printed_toc(pages):
    entries = []
    for page in pages:
        labelled = []
        for line in page.get('lines',[]):
            match = re.match(r'^(.+?)\s*(?:\.\s*){3,}\s*(\d+)\s*$',line['text'])
            if not match:
                continue
            target = next((link for link in page['links'] if link.get('target_page') is not None and
                           intersection(line['bbox'],link['bbox'])/max(1,area(line['bbox'])) > .3),None)
            if target:
                labelled.append({'title':match.group(1).strip(),'printed_page':int(match.group(2)),
                                 'target_page':int(target['target_page'])+1,'line_id':line['line_id'],
                                 'link_id':target['link_id'],'bbox':line['bbox'],'toc_page':page['page'],
                                 'level':1,'source':'printed_toc_link'})
        if len(labelled) >= 6 and len({e['target_page'] for e in labelled}) >= 3:
            page['toc_entries'] = labelled
            entries.extend(labelled)
    return entries


def furniture_key(text):
    text = re.sub(r'\s+', ' ', text).strip().casefold()
    # Varying section/item numbers are content. Only normalize an actual page
    # counter, never every digit in a sentence ("Lesson 1" != "Lesson 2").
    if re.fullmatch(r'(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?', text):
        return re.sub(r'\d+', '#', text)
    return text


def line_is_furniture(line, page, repeated):
    y = line['bbox'][1]/page['height']
    text = re.sub(r'\s+', ' ', line['text']).strip()
    key = furniture_key(text)
    return (y < .125 or y > .91) and (key in repeated or bool(re.fullmatch(r'\d+(?:\s+(?:of|/)\s+\d+)?', text)))


def heading_matches(pages, printed_entries, body_size):
    result = {}
    by_page = defaultdict(list)
    for node in printed_entries:
        if node[1].strip():
            by_page[int(node[2])].append(node)
    # Printed navigation matches one source line on its destination page.
    for page in pages:
        available = [l for l in page.get('lines', []) if not l.get('furniture')]
        used = set()
        for level, title, _ in by_page[page['page']]:
            key = text_key(title)
            candidates = []
            for line in available:
                candidate = re.sub(r'^\d+(?:\.\d+)*\s*', '', line['text'].strip())
                score = 2 if text_key(candidate) == key else 1 if text_key(line['text']) == key else 0
                if score and line['line_id'] not in used:
                    candidates.append((score, line['size'], -line['bbox'][1], line))
            if candidates:
                line = max(candidates, key=lambda item:item[:3])[3]
                used.add(line['line_id'])
                result[line['line_id']] = {'level':min(6,max(1,int(level))), 'title':title,
                                           'heading_method':'printed_toc_link'}
    sizes = sorted({round(l['size'], 1) for p in pages for l in p.get('lines', [])
                    if l['size'] >= body_size*1.15 and not l.get('furniture')}, reverse=True)
    for page in pages:
        for line in page.get('lines', []):
            if line.get('furniture') or line['line_id'] in result:
                continue
            text = line['text'].strip()
            ratio = line['size']/body_size
            short = len(text) < 160 and len(text.split()) < 18
            logo_glyph = ratio > 3 and len(text.split()) <= 1 and len(text) < 4
            shortcut_sequence = text.startswith('[') and text.count('[') >= 3
            if short and not logo_glyph and not shortcut_sequence and (ratio >= 1.2 or ratio >= 1.07 and line['bold_ratio'] > .9) and not re.search(r'[.!?;,]$', text):
                rank = next((i for i,s in enumerate(sizes) if s <= line['size']+.15), len(sizes))
                result[line['line_id']] = {'level':min(4,rank+1), 'title':text,
                                           'heading_method':'typography', 'confidence':.75}
    return result


def physical_rows(lines):
    rows = []
    for line in sorted(lines, key=lambda l:(l['bbox'][1],l['bbox'][0])):
        center = (line['bbox'][1]+line['bbox'][3])/2
        if rows and abs(center-rows[-1]['center']) <= max(2.0, line['size']*.28):
            rows[-1]['lines'].append(line)
        else:
            rows.append({'center':center, 'lines':[line]})
    for row in rows:
        row['lines'].sort(key=lambda l:l['bbox'][0])
    return rows


def infer_tables(lines, heading_ids):
    """Conservative aligned-cell inference, not a per-software table parser."""
    rows = physical_rows(lines)
    candidates = []
    for row in rows:
        cells = []
        for line in row['lines']:
            # A single PDF line can contain multiple widely spaced cells.
            groups = []
            for run in line['runs']:
                if groups and run['bbox'][0]-groups[-1][-1]['bbox'][2] > max(20, line['size']*2.5):
                    groups.append([])
                elif not groups:
                    groups.append([])
                groups[-1].append(run)
            for runs in groups:
                if plain(runs).strip():
                    cells.append({'runs':runs, 'bbox':box_union([r['bbox'] for r in runs]), 'line_ids':[line['line_id']]})
        cells.sort(key=lambda c:c['bbox'][0])
        # Merge genuinely contiguous line fragments (e.g. individual words).
        merged = []
        for cell in cells:
            if merged and cell['bbox'][0]-merged[-1]['bbox'][2] < max(3, statistics.median(l['size'] for l in row['lines'])*.55):
                merged[-1]['runs'].append({'text':' ', 'synthetic':'cell_join'})
                merged[-1]['runs'].extend(cell['runs'])
                merged[-1]['bbox'] = box_union([merged[-1]['bbox'],cell['bbox']])
                merged[-1]['line_ids'].extend(cell['line_ids'])
            else:
                merged.append(cell)
        valid = len(merged) >= 2 and not any(l['line_id'] in heading_ids for l in row['lines'])
        valid = valid and all(len(plain(c['runs'])) < 220 for c in merged[:1])
        candidates.append({**row, 'cells':merged, 'valid':valid})
    blocks, taken, i = [], set(), 0
    while i < len(candidates):
        first = candidates[i]
        if not first['valid']:
            i += 1; continue
        group = [first]
        j = i+1
        starts = [c['bbox'][0] for c in first['cells']]
        while j < len(candidates):
            row = candidates[j]
            previous = group[-1]
            spacing = row['center']-previous['center']
            if spacing > max(26, statistics.median(l['size'] for l in first['lines'])*2.4):
                break
            aligned = row['valid'] and len(row['cells']) == len(starts) and all(abs(c['bbox'][0]-x) < 10 for c,x in zip(row['cells'],starts))
            if aligned:
                group.append(row); j += 1; continue
            # Preserve description continuation lines within an established
            # table; do not absorb a following one-column paragraph.
            if len(group) >= 3 and len(row['cells']) == 1 and abs(row['cells'][0]['bbox'][0]-starts[-1]) < 10 and spacing < 18:
                cell = group[-1]['cells'][-1]
                cell['runs'].append({'text':'\n', 'synthetic':'cell_continuation'})
                cell['runs'].extend(row['cells'][0]['runs'])
                cell['line_ids'].extend(row['cells'][0]['line_ids'])
                group[-1]['lines'].extend(row['lines'])
                group[-1]['center'] = row['center']
                j += 1; continue
            break
        # Several aligned long prose columns are columns, not a data table.
        short_left = sum(len(plain(r['cells'][0]['runs']).split()) <= 8 for r in group)
        column_prose = False
        if len(starts) == 2:
            gap = starts[1]-starts[0]
            occupancy = statistics.median((r['cells'][0]['bbox'][2]-r['cells'][0]['bbox'][0])/max(1,gap) for r in group)
            column_prose = occupancy > .58
        if len(group) >= 3 and short_left/len(group) >= .75 and not column_prose:
            owned = list(dict.fromkeys(l['line_id'] for r in group for l in r['lines']))
            taken.update(owned)
            all_lines = [l for r in group for l in r['lines']]
            header = all(sum(len(x.get('text','')) for x in c['runs'] if x.get('flags',0)&16) /
                         max(1,len(plain(c['runs']))) > .8 for c in first['cells'])
            blocks.append({'type':'table', 'page':all_lines[0]['page'], 'bbox':box_union([l['bbox'] for l in all_lines]),
                           'line_ids':owned, 'rows':[{'cells':r['cells']} for r in group], 'header':header,
                           'detection':'aligned_cells', 'confidence':.8})
            i = j
        else:
            i += 1
    return blocks, taken


def ruled_tables(page, lines, heading_ids):
    blocks, taken, warnings = [], set(), []
    try:
        found = page.find_tables().tables
        for table in found:
            cells = []
            owned = []
            for row in table.rows:
                result = []
                for cell_box in row.cells:
                    cell_lines = [l for l in lines if cell_box and intersection(l['bbox'],cell_box)/max(1,area(l['bbox'])) > .55]
                    owned.extend(cell_lines)
                    result.append({'bbox':list(cell_box) if cell_box else None, 'runs':join_lines(cell_lines),
                                   'line_ids':[l['line_id'] for l in cell_lines]})
                cells.append({'cells':result})
            ids = list(dict.fromkeys(l['line_id'] for l in owned))
            if ids and not any(x in heading_ids or x in taken for x in ids):
                blocks.append({'type':'table','page':lines[0]['page'], 'bbox':list(table.bbox), 'line_ids':ids,
                               'rows':cells,'header':bool(table.header and not table.header.external),
                               'detection':'ruled_geometry','confidence':.95})
                taken.update(ids)
    except Exception as exc:
        warnings.append({'code':'table_detection_failed','error_type':type(exc).__name__})
    return blocks, taken, warnings


def reading_order(blocks, width):
    """Read separated prose columns top-to-bottom, with full-width separators."""
    blocks = sorted(blocks, key=lambda b:(b['bbox'][1],b['bbox'][0]))
    if len(blocks) < 4:
        return blocks
    # A persistent geometric gutter separates columns. Tables remain atomic.
    for fraction in (.5,.45,.55):
        cut = width*fraction
        left = [b for b in blocks if b['bbox'][2] <= cut-5]
        right = [b for b in blocks if b['bbox'][0] >= cut+5]
        if len(left) >= 2 and len(right) >= 2:
            overlaps = sum(any(min(a['bbox'][3],b['bbox'][3])-max(a['bbox'][1],b['bbox'][1]) > 0 for b in right) for a in left)
            if overlaps >= 2:
                spans = [b for b in blocks if b not in left and b not in right]
                ordered, rest = [], list(blocks)
                for separator in spans:
                    before = [b for b in rest if b['bbox'][3] <= separator['bbox'][1]]
                    ordered.extend(sorted([b for b in before if b in left], key=lambda b:b['bbox'][1]))
                    ordered.extend(sorted([b for b in before if b in right], key=lambda b:b['bbox'][1]))
                    ordered.append(separator)
                    rest = [b for b in rest if b not in before and b is not separator]
                ordered.extend(sorted([b for b in rest if b in left], key=lambda b:b['bbox'][1]))
                ordered.extend(sorted([b for b in rest if b in right], key=lambda b:b['bbox'][1]))
                return ordered
    return blocks


def compose_overlapping_images(page, blocks, assets_dir):
    """Collapse intersecting raster fragments into faithful rendered regions.

    Original resources and placement records remain in the image sidecar.
    This also preserves text/vector overlays within the raster region.
    """
    images = [b for b in blocks if b['type'] == 'image']
    remaining = list(images)
    replacements = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if any(intersection(candidate['bbox'], other['bbox']) > 0 for other in group):
                    group.append(candidate);remaining.remove(candidate);changed = True
        if len(group) == 1:
            replacements.extend(group)
            continue
        bbox = box_union([b['bbox'] for b in group])
        clip = fitz.Rect(bbox) & page.rect
        if clip.is_empty:
            replacements.extend(group)
            continue
        scale = min(2,math.sqrt(4_000_000/max(1,clip.width*clip.height)))
        binary = render_png(page, matrix=fitz.Matrix(scale,scale),clip=clip)
        filename = digest(binary)+'.png';(assets_dir/filename).write_bytes(binary)
        replacements.append({'type':'image','page':group[0]['page'],'bbox':bbox,'line_ids':[],
                             'image_id':'region_'+digest('|'.join(b['image_id'] for b in group))[:12],
                             'filename':filename,'rendered_composite':True,
                             'source_image_ids':[b['image_id'] for b in group]})
    return [b for b in blocks if b['type'] != 'image']+replacements


def visual_preservation_reasons(page, snapshot):
    """Find layout evidence that a text/raster projection cannot represent.

    These signals select supplementary page pixels, never rewrite source text
    or classify a page by the software or document name.
    """
    reasons = []
    # The sorted text projection can omit unmapped glyphs; original span text
    # is a second independent signal and remains preserved in the snapshot.
    span_text = ''.join(line['text'] for line in snapshot['lines'])
    if (fallback.text_quality(snapshot.get('raw_text','')).get('invalid_character_count')
            or fallback.text_quality(span_text).get('invalid_character_count')):
        reasons.append('unmapped_text_glyphs')
    math_characters = 0
    for line in snapshot['lines']:
        for run in line['runs']:
            if re.search(r'^(?:CM(?:MI|SY|EX|R)|STIX|Math)|symbol|math',run.get('font',''),re.I):
                math_characters += sum(not c.isspace() and c not in '▷▸▶►→←•' for c in run['text'])
    if math_characters >= 20:
        reasons.append('mathematical_layout')
    drawings = fallback.decode_call(page.get_drawings)
    graphic_paths = []
    graphic_items = 0
    for drawing in drawings:
        original = drawing['rect']
        # A horizontal/vertical stroke has a zero-area geometry rectangle, but
        # can still be an authored connector or diagram. Include its stroke.
        padding = max(.5, float(drawing.get('width') or 0) / 2)
        rect = fitz.Rect(original.x0-padding, original.y0-padding,
                         original.x1+padding, original.y1+padding) & page.rect
        if rect.is_empty or max(rect.width, rect.height) < 2:
            continue
        # Original raster bytes do not include PDF vector overlays. Even one
        # authored path can carry instructions, so path counts cannot decide
        # whether a visual is disposable.
        graphic_paths.append(rect)
        graphic_items += len(drawing.get('items', []))
    if graphic_paths:
        reasons.append('vector_graphics')
    if len(graphic_paths) >= 8 or graphic_items >= 24:
        reasons.append('vector_graphics_or_complex_layout')
    return reasons


def scoped_visuals(page, snapshot, assets_dir):
    """Preserve graphics only inside the original selected chapter bands."""
    assets, blocks = [], []
    for index, (low, high) in enumerate(snapshot.get('scope_regions', []), 1):
        clip = fitz.Rect(0, low, page.rect.width, high) & page.rect
        if clip.is_empty:
            continue
        scale = min(1.5, math.sqrt(4_000_000 / max(1, clip.width * clip.height)))
        binary = render_png(page, matrix=fitz.Matrix(scale, scale), clip=clip)
        name = digest(binary) + '.png'
        (assets_dir / name).write_bytes(binary)
        image_id = f'SCOPE{snapshot["page"]}_{index}'
        assets.append({'image_id': image_id, 'path': str(assets_dir / name), 'bbox': list(clip),
                       'page': snapshot['page'], 'role': 'selected_chapter_visual', 'status': 'saved',
                       'sha256': digest(binary), 'bytes': len(binary), 'original_binary': False})
        blocks.append({'type': 'page_image', 'page': snapshot['page'], 'bbox': list(clip), 'line_ids': [],
                       'image_id': image_id, 'filename': name, 'supplemental_visual': True,
                       'text': '[Original visual of the selected chapter region.]'})
    return assets, blocks


def heading_prefix_matches(previous, following):
    """Bind split heading labels only when their original layout supports it."""
    label = previous['text'].strip()
    title = following['text'].strip()
    # Another numbered/named heading starts a new section, even when nearby.
    if (re.match(r'^\d+(?:\.\d+)*\.?(?:\s|$)', title)
            or re.match(r'^(?:Chapter|Section|Part)\s+', title, re.I)):
        return False
    before, after = previous['bbox'], following['bbox']
    if re.fullmatch(r'\d+(?:\.\d+)*\.?', label):
        horizontal_gap = after[0] - before[2]
        return (previous['block_index'] == following['block_index']
                and abs(before[1]-after[1]) < 3
                and 0 <= horizontal_gap <= max(10, previous['size']*4))
    if re.fullmatch(r'(?:Chapter|Section|Part)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])', label, re.I):
        gap = after[1]-before[3]
        # A larger title aligned below a smaller label is one authored chapter
        # heading. Equal/smaller typography may be a following sibling/subsection.
        return (0 <= gap < 50 and abs(before[0]-after[0]) < 3
                and following['size'] > previous['size']*1.05)
    return False


def structured_page(page, snapshot, headings, body_size, assets_dir, document, image_cache):
    lines = snapshot['lines']
    lines_by_id = {line['line_id']: line for line in lines}
    if len(lines) > 100_000:
        raise ValueError('page_line_budget_exceeded')
    annotations = [make_block('annotation', [l], role='running_furniture') for l in lines if l.get('furniture')]
    content = [l for l in lines if not l.get('furniture')]
    toc_lines = {e['line_id'] for e in snapshot.get('toc_entries',[])}
    toc_block = make_block('toc',[l for l in content if l['line_id'] in toc_lines],entries=snapshot.get('toc_entries',[])) if toc_lines else None
    content = [l for l in content if l['line_id'] not in toc_lines]
    blocks, taken, warnings = fallback.decode_call(ruled_tables, page, content, headings)
    unruled, owned = infer_tables([l for l in content if l['line_id'] not in taken], headings)
    blocks.extend(unruled); taken.update(owned)
    if toc_block:
        blocks.append(toc_block)
    pending, current_kind = [], None
    def flush():
        nonlocal pending
        if not pending:
            return
        if current_kind == 'list':
            items = []
            base = min(l['bbox'][0] for l in pending)
            for line in pending:
                marker = re.match(r'^\s*([•●▪◦‣\-–]|\d+[.)]|[a-zA-Z][.)])\s+', line['text'])
                if marker or not items:
                    runs = list(line['runs'])
                    if marker:
                        # Keep authored marker in source spans; renderer adds
                        # its Markdown marker rather than repeating the glyph.
                        remaining = marker.end()
                        trimmed = []
                        for run in runs:
                            n = min(remaining,len(run['text'])); remaining -= n
                            if n < len(run['text']):
                                trimmed.append({**run,'text':run['text'][n:]})
                        runs = trimmed
                    items.append({'marker':marker.group(1) if marker and marker.group(1)[0].isdigit() else '-',
                                  'depth':max(0,round((line['bbox'][0]-base)/18)), 'runs':runs,
                                  'line_ids':[line['line_id']]})
                else:
                    items[-1]['runs'].append({'text':' ','synthetic':'list_continuation'})
                    items[-1]['runs'].extend(line['runs']); items[-1]['line_ids'].append(line['line_id'])
            blocks.append(make_block('list', pending, items=items))
        else:
            blocks.append(make_block(current_kind, pending))
        pending = []
    for line in sorted(content,key=lambda l:(l['block_index'],l['bbox'][1],l['bbox'][0])):
        if line['line_id'] in taken:
            flush(); continue
        if line['line_id'] in headings:
            prefix = []
            if pending:
                last = pending[-1]
                if heading_prefix_matches(last, line):
                    prefix = [pending.pop()]
            flush()
            # A number/label can itself meet the typography heading threshold,
            # so it may already have been emitted instead of remaining pending.
            # Bind it to its adjacent authored title using original geometry.
            if blocks and blocks[-1]['type'] == 'heading':
                previous = blocks[-1]
                last_line = lines_by_id[previous['line_ids'][-1]]
                if len(previous['line_ids']) == 1 and heading_prefix_matches(last_line, line):
                    blocks.pop()
                    prefix = [lines_by_id[line_id] for line_id in previous['line_ids']] + prefix
            heading = dict(headings[line['line_id']])
            if prefix:
                heading['title'] = ' '.join(part['text'].strip() for part in [*prefix,line])
            blocks.append(make_block('heading', [*prefix,line], **heading))
            current_kind = None; continue
        marker = re.match(r'^\s*([•●▪◦‣\-–]|\d+[.)]|[a-zA-Z][.)])\s+',line['text'])
        kind = 'code' if line['mono_ratio'] > .65 else 'list' if marker else 'paragraph'
        # A wrapped list item can switch to prose typography without ending it.
        if pending and current_kind == 'list' and not marker and line['block_index'] == pending[-1]['block_index']:
            kind = 'list'
        if line['size'] < body_size*.85 and line['bbox'][1] > snapshot['height']*.75:
            kind = 'footnote'
        if pending:
            last = pending[-1]
            gap = line['bbox'][1]-last['bbox'][3]
            same_column = abs(line['bbox'][0]-last['bbox'][0]) < 22 or kind == 'list'
            if kind != current_kind or gap > max(7,line['size']*.65) or not same_column or line['block_index'] != last['block_index']:
                flush()
        current_kind = kind; pending.append(line)
    flush()
    images = []
    # A shared embedded image can span a selected and an unselected chapter.
    # Scoped extraction therefore preserves clipped page regions below instead
    # of exporting the unrestricted original image binary.
    image_infos = [] if 'scope_regions' in snapshot else fallback.decode_call(page.get_image_info, xrefs=True)
    for i, info in enumerate(image_infos,1):
        xref = info.get('xref',0)
        image_id = f'I{snapshot["page"]}_{i}'
        image = {'image_id':image_id,'page':snapshot['page'],'bbox':list(info['bbox']),
                 'xref':xref,'width':info['width'],'height':info['height'],'role':'unknown'}
        try:
            if xref and xref in image_cache:
                image.update(image_cache[xref])
            else:
                extracted = fallback.decode_call(document.extract_image, xref) if xref else None
                if extracted and extracted.get('image'):
                    binary, extension = extracted['image'],extracted['ext']
                    name = digest(binary)+'.'+extension
                    (assets_dir/name).write_bytes(binary)
                    resource = {'filename':name,'sha256':digest(binary),'bytes':len(binary),'status':'saved','original_binary':True}
                    if extracted.get('smask'):
                        base = fallback.decode_call(fitz.Pixmap, document,xref)
                        mask = fallback.decode_call(fitz.Pixmap, document,extracted['smask'])
                        if base.alpha:
                            base = fallback.decode_call(fitz.Pixmap, base,0)
                        combined = fallback.decode_call(fitz.Pixmap, base,mask)
                        preview = fallback.decode_call(combined.tobytes, 'png')
                        preview_name = digest(preview)+'.png'
                        (assets_dir/preview_name).write_bytes(preview)
                        resource['display_filename'] = preview_name
                    image.update(resource)
                    if xref:
                        image_cache[xref] = resource
                else:
                    clip = fitz.Rect(info['bbox']) & page.rect
                    if clip.is_empty:
                        raise ValueError('invalid_image_bbox')
                    scale = min(1.5,math.sqrt(4_000_000/max(1,clip.width*clip.height)))
                    binary = render_png(page, matrix=fitz.Matrix(scale,scale),clip=clip)
                    name = digest(binary)+'.png'; (assets_dir/name).write_bytes(binary)
                    image.update(filename=name,sha256=digest(binary),bytes=len(binary),status='saved',original_binary=False)
            blocks.append({'type':'image','page':snapshot['page'],'bbox':image['bbox'],'line_ids':[],
                           'image_id':image_id,'filename':image.get('display_filename',image['filename'])})
        except OSError:
            raise
        except Exception as exc:
            image.update(status='missing',reason=type(exc).__name__)
            warnings.append({'code':'image_extraction_failed','image_id':image_id,'error_type':type(exc).__name__})
        images.append(image)
    if any(image.get('status') != 'saved' for image in images):
        # Recover the whole page through independent text/rendering engines.
        # Merely recording a warning can otherwise turn an image-only page
        # into a successful blank page, or silently omit a figure beside text.
        raise ValueError('embedded_image_requires_page_recovery')
    try:
        blocks = compose_overlapping_images(page,blocks,assets_dir)
    except OSError:
        raise
    except Exception as exc:
        warnings.append({'code':'image_composition_failed','error_type':type(exc).__name__})
    assets = []
    visual_degraded = False
    try:
        if 'scope_regions' in snapshot:
            assets, region_blocks = scoped_visuals(page, snapshot, assets_dir)
            blocks.extend(region_blocks)
        else:
            visual_reasons = visual_preservation_reasons(page,snapshot)
        if 'scope_regions' not in snapshot and visual_reasons:
            asset = fallback.preserve_page_visual(Path(document.name),page.number,
                pymupdf_page=page,output_dir=assets_dir.parent,warnings=warnings)
            if asset:
                asset.update(role='supplemental_page_visual',reasons=visual_reasons)
                assets.append(asset)
                blocks.append({'type':'page_image','page':snapshot['page'],'bbox':asset['bbox'],'line_ids':[],
                    'image_id':asset['image_id'],'filename':Path(asset['path']).name,
                    'supplemental_visual':True,'reasons':visual_reasons,
                    'text':'[Original page visual retained alongside structured text for graphics, formulas, or glyph fidelity.]'})
            else:
                warnings.append({'code':'supplemental_visual_unavailable','reasons':visual_reasons})
                visual_degraded = True
    except OSError:
        raise
    except Exception as exc:
        warnings.append({'code':'visual_inspection_failed','error_type':type(exc).__name__})
        visual_degraded = True
    blocks = reading_order(blocks,snapshot['width'])
    # Adjacent authored bullets are one list, even when the PDF producer used
    # a separate text object for each item. A heading/figure remains a boundary.
    combined = []
    for block in blocks:
        if combined and block['type'] == combined[-1]['type'] == 'list' and block['bbox'][1]-combined[-1]['bbox'][3] < 22:
            last = combined[-1]
            last['items'].extend(block['items']);last['line_ids'].extend(block['line_ids'])
            last['runs'].extend([{'text':'\n','synthetic':'list_item_join'},*block['runs']])
            last['text']=plain(last['runs']);last['bbox']=box_union([last['bbox'],block['bbox']])
        else:
            combined.append(block)
    blocks = combined
    # Assertions are content conservation gates, not aesthetic confidence.
    counts = Counter(x for b in [*blocks,*annotations] for x in b['line_ids'])
    missing = [l for l in lines if not counts[l['line_id']]]
    if missing:
        blocks.extend(make_block('paragraph',[line],recovery='unassigned_line') for line in missing)
        warnings.append({'code':'residual_lines_retained','count':len(missing)})
        counts.update(l['line_id'] for l in missing)
    if any(n != 1 for n in counts.values()):
        raise ValueError('duplicate_line_assignment')
    return {'mode':'structured','blocks':blocks,'annotations':annotations,'images':images,'assets':assets,
            'warnings':warnings,'degraded':visual_degraded,
            'capture_status':'partial_original_preserved' if visual_degraded else 'saved',
            'line_count':len(lines),'assigned_line_count':sum(counts.values()),
            'coverage':{'every_line_assigned_once':len(counts)==len(lines) and all(n==1 for n in counts.values()),
                        'raw_text_saved':True,'raw_layout_saved':True}}


def make_parts(pages, source_id, soft_limit):
    page_modes = {page['page']: page['mode'] for page in pages}
    sections, stack = [], []
    current = {'title':'Document beginning','path':[], 'level':0,'blocks':[]}
    for page in pages:
        for block in page['blocks']:
            if block['type'] == 'heading':
                if current['blocks']:
                    sections.append(current)
                level = block['level']
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level,block.get('title') or block['text'].strip()))
                current = {'title':stack[-1][1], 'path':[x[1] for x in stack], 'level':level,'blocks':[block]}
            else:
                current['blocks'].append(block)
        if not page['blocks']:
            current['blocks'].append({'type':'blank_page','page':page['page'],'line_ids':[],
                                     'text':'[Blank page; original page retained.]','bbox':None})
    if current['blocks']:
        sections.append(current)
    parts = []
    for section_index,section in enumerate(sections,1):
        groups, pending, size = [], [], 0
        for block in section['blocks']:
            rendered = block_markdown(block)
            if pending and size+len(rendered)+2 > soft_limit:
                groups.append(pending); pending=[];size=0
            pending.append(block);size += len(rendered)+2
        if pending:
            groups.append(pending)
        for chunk,blocks in enumerate(groups,1):
            content = '\n\n'.join(block_markdown(b) for b in blocks if block_markdown(b))
            page_numbers = sorted({b['page'] for b in blocks})
            part_id = 'part_'+digest(source_id+'|'+str(section_index)+'|'+str(chunk))[:20]
            ranges,offset = [],0
            for block in blocks:
                text = block_markdown(block)
                if not text:
                    continue
                if ranges:
                    offset += 2
                ranges.append({'block_type':block['type'],'page':block['page'],'start_offset':offset,
                               'end_offset':offset+len(text),'line_ids':block.get('line_ids',[])})
                offset += len(text)
            parts.append({'part_id':part_id,'source_id':source_id,'kind':'pdf_section',
                          'label':section['title']+(f' ({chunk})' if len(groups)>1 else ''),'ordinal':len(parts)+1,
                          'content':content,'content_sha256':digest(content),'char_count':len(content),
                          'locator':{'page_start':min(page_numbers),'page_end':max(page_numbers),'pages':page_numbers,
                                     'heading_path':section['path'],'heading_level':section['level'],
                                     'section_index':section_index,'chunk':chunk},
                          'metadata':{'blocks':blocks,'block_ranges':ranges,'soft_limit_exceeded':len(content)>soft_limit,
                                      'extraction_modes':sorted({page_modes[n] for n in page_numbers})}})
    return parts


def _close_document(document):
    if not document.is_closed:
        document.close()


def document_structure(snapshots, outline=None):
    """Infer structure from page content; legacy bookmark input is ignored."""
    repeated = furniture_keys(snapshots)
    font_weights = Counter()
    for snapshot in snapshots:
        if 'layout_error' in snapshot:
            continue
        for line in snapshot['lines']:
            line['furniture'] = line_is_furniture(line,snapshot,repeated)
            if not line['furniture']:
                for run in line['runs']:
                    if not run['flags'] & 16 and math.isfinite(run['size']) and run['size'] > 0:
                        font_weights[round(run['size'],1)] += len(run['text'].strip())
    body_size = font_weights.most_common(1)[0][0] if font_weights else 11
    body_size = max(.1, body_size)
    toc = printed_toc(snapshots)
    printed_entries = [[e['level'],e['title'],e['target_page']] for e in toc]
    headings = heading_matches(snapshots,printed_entries,body_size)
    return body_size, toc, headings


def extract_pdf(path, output_dir, source_id=None, password=None, soft_limit=10_000, fail_pages=None, extraction_scope=None):
    """Materialize structured parts and recovery assets in an isolated directory.

    The soft limit applies between complete blocks. Parsing failures degrade at
    page granularity; file-system failures and cancellation propagate. The
    optional ``fail_pages`` hook injects one-based page failures for validation.
    """
    with ExitStack() as resources:
        return _extract_pdf(path, output_dir, source_id, password, soft_limit, fail_pages, resources, extraction_scope)


def _extract_pdf(path, output_dir, source_id, password, soft_limit, fail_pages, resources, extraction_scope=None):
    path, output_dir = filesystem_path(path), filesystem_path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    assets_dir = output_dir/'assets'; assets_dir.mkdir(exist_ok=True)
    raw_dir = output_dir/'raw';raw_dir.mkdir(exist_ok=True)
    source = raw_dir/'source.pdf'
    if path != source:
        shutil.copyfile(path,source)
    raw_sha = file_digest(source)
    source_id = source_id or 'SRC-'+raw_sha[:12]
    document = None
    opening_error = None
    try:
        if fitz is None:
            raise ImportError('pymupdf unavailable')
        document = fallback.decode_call(fitz.open, path)
        resources.callback(_close_document, document)
        if document.needs_pass and not document.authenticate(password or ''):
            raise ValueError('pdf_password_required_or_invalid')
        if document.page_count < 1:
            raise ValueError('pdf_has_no_pages')
        if document.page_count > 5000:
            raise ValueError('page_count_safety_budget_exceeded')
    except OSError:
        raise
    except Exception as exc:
        opening_error = {'code':'pdf_open_failed','error_type':type(exc).__name__,'message':str(exc)[:300]}
        if document:
            document.close();document=None
    snapshots,pages = [],[]
    is_scoped = bool(extraction_scope and extraction_scope.get('mode') == 'selected')
    original_page_count = document.page_count if document else None
    if is_scoped and document is None:
        raise ValueError('Selected PDF chapters require the original geometry parser; full-document recovery is not allowed')
    from .scope import page_regions, scoped_snapshot
    regions = page_regions(extraction_scope, original_page_count) if is_scoped else {}
    if is_scoped and not regions:
        raise ValueError('Selected PDF chapters contain no valid page regions')
    outline = []
    document_warnings = []
    if document:
        for number in (sorted(regions) if is_scoped else range(1, len(document) + 1)):
            index = number - 1
            try:
                snapshot = fallback.decode_call(raw_page, document[index],number)
                snapshots.append(scoped_snapshot(snapshot, regions[number]) if is_scoped else snapshot)
            except OSError:
                raise
            except Exception as exc:
                if is_scoped:
                    raise ValueError(f'Cannot safely read selected PDF chapter on original page {number}') from exc
                snapshots.append({'page':index+1,'lines':[],'links':[],
                                  'layout_error':{'error_type':type(exc).__name__,'message':str(exc)[:300]}})
        analysis_error = None
        try:
            body_size, toc, headings = fallback.decode_call(document_structure, snapshots)
        except OSError:
            raise
        except Exception as exc:
            analysis_error = {'code':'document_structure_unavailable','error_type':type(exc).__name__}
            document_warnings.append(analysis_error)
            body_size, toc, headings = None, [], {}
        image_cache = {}
        for snapshot in snapshots:
            number = snapshot['page']
            index = number - 1
            save_json(raw_dir/f'page-{number:04d}.json',snapshot)
            (raw_dir/f'page-{number:04d}.txt').write_text(snapshot.get('raw_text',''),encoding='utf-8')
            primary_page = None
            try:
                primary_page = document[index]
                if analysis_error:
                    raise ValueError('document_structure_unavailable')
                if index+1 in (fail_pages or []):
                    raise RuntimeError('injected_structure_failure')
                if 'layout_error' in snapshot:
                    raise ValueError('page_layout_unavailable')
                quality = fallback.text_quality(snapshot.get('raw_text',''))
                if quality.get('unreadable_glyphs'):
                    raise ValueError('unreadable_text_glyphs')
                result = structured_page(primary_page,snapshot,headings,body_size,assets_dir,document,image_cache)
                if not any(b['type'] not in {'annotation'} for b in result['blocks']):
                    # An authored blank/header-only page is a valid structural
                    # result. A vector-only page requires a visual asset.
                    if fallback.decode_call(primary_page.get_drawings):
                        raise ValueError('vector_only_body_requires_visual')
                    result['blocks']=[{'type':'blank_page','page':index+1,'bbox':None,'line_ids':[],
                                       'text':'[No body content; original page and annotations retained.]'}]
                    result['content_kind']='annotation_only' if snapshot['lines'] else 'blank_page'
            except OSError:
                raise
            except Exception as exc:
                if is_scoped:
                    # Recovery must obey the same boundary. Alternate engines
                    # read full pages and would reintroduce unselected chapters.
                    blocks = [make_block('paragraph', [line], recovery='selected_region') for line in snapshot['lines']]
                    warnings = [{'code': 'selected_region_recovery', 'error_type': type(exc).__name__}]
                    assets = []
                    try:
                        assets, visual_blocks = scoped_visuals(primary_page, snapshot, assets_dir)
                        blocks.extend(visual_blocks)
                    except OSError:
                        raise
                    except Exception as visual_error:
                        warnings.append({'code': 'selected_visual_unavailable', 'error_type': type(visual_error).__name__})
                    result = {'mode': 'selected_region_recovery', 'blocks': blocks, 'assets': assets,
                              'warnings': warnings, 'degraded': True, 'capture_status': 'saved' if blocks else 'missing'}
                else:
                    result = fallback.recover_page(path,index,pymupdf_page=primary_page,password=password,
                                                   reason=type(exc).__name__+': '+str(exc)[:160],output_dir=output_dir,include_visual=True)
                result.setdefault('annotations',[]);result.setdefault('images',[])
                result.setdefault('line_count',len(snapshot['lines']))
                result.setdefault('assigned_line_count',len(snapshot['lines']))
                result.setdefault('coverage',{'raw_layout_saved':True,'raw_text_saved':True,'structured_assignment_available':False})
                for block in result['blocks']:
                    block.setdefault('page',index+1);block.setdefault('line_ids',[])
                    asset = next((a for a in result.get('assets',[]) if a.get('image_id') == block.get('image_id')),None)
                    if asset:
                        block['filename']=Path(asset['path']).name
            pages.append({'page':index+1,'width':snapshot.get('width'),'height':snapshot.get('height'),
                          'raw_snapshot':f'raw/page-{index+1:04d}.json','links':snapshot['links'],**result})
        document.close()
    else:
        # A missing/broken primary engine must not reduce a readable multipage
        # PDF to page one. Discover its page tree independently before recovery.
        independent_count = None
        try:
            from pypdf import PdfReader
            with PdfReader(path) as reader:
                if reader.is_encrypted and not reader.decrypt(password or ''):
                    raise ValueError('pdf_password_required_or_invalid')
                independent_count = len(reader.pages)
                if independent_count > 5000:
                    independent_count = None
        except OSError:
            raise
        except Exception:
            pass
        for index in range(independent_count or 1):
            result = (fallback.recover_page(path,index,password=password,reason=opening_error,output_dir=output_dir,include_visual=True)
                      if independent_count else fallback.recover_document(path,reason=opening_error,output_dir=output_dir))
            result.setdefault('annotations',[]);result.setdefault('images',[])
            for block in result['blocks']:
                block.setdefault('page',index+1);block.setdefault('line_ids',[])
                asset = next((a for a in result.get('assets',[]) if a.get('image_id') == block.get('image_id')),None)
                if asset:
                    block['filename']=Path(asset['path']).name
            pages.append({'page':index+1,'links':[],'page_number_known':independent_count is not None,**result})
        body_size = None
        toc = []
    page_count_known = all(p.get('page_number_known',True) for p in pages)
    parts = make_parts(pages,source_id,max(1,int(soft_limit)))
    for part in parts:
        part_dir=output_dir/'parts';part_dir.mkdir(exist_ok=True)
        (part_dir/(part['part_id']+'.md')).write_text(part['content'],encoding='utf-8')
    result = {'schema_version':VERSION,'source_id':source_id,'source_name':path.name,'source_sha256':raw_sha,
              'source_pdf':'raw/source.pdf','outline':outline,'printed_toc':toc,'body_font_size':body_size,
              'pages':pages,'parts':parts,'opening_error':opening_error,'warnings':document_warnings,
              'images':[image for page in pages for image in page.get('images',[])],
              'assets':[asset for page in pages for asset in page.get('assets',[])],
              'links':[link for page in pages for link in page.get('links',[])],
              'summary':{'pages':len(pages) if page_count_known else None,
                         'original_page_count':original_page_count or (len(pages) if page_count_known else None),
                         'selected_page_numbers':[p['page'] for p in pages] if is_scoped else [],
                         'extraction_mode':'selected' if is_scoped else 'full',
                         'page_count_known':page_count_known,'records':len(pages),
                         'parts':len(parts),'modes':dict(Counter(p['mode'] for p in pages)),
                         'blocks':dict(Counter(b['type'] for p in pages for b in p['blocks'])),
                         'all_pages_represented':({p['page'] for p in pages} <= {n for p in parts for n in p['locator']['pages']}) if page_count_known else None,
                         'structured_line_coverage':all(p['coverage']['every_line_assigned_once'] for p in pages if p['mode']=='structured'),
                         'degraded_pages':[p['page'] for p in pages if p['degraded']],
                         'soft_limit_exceeded_parts':sum(p['metadata']['soft_limit_exceeded'] for p in parts)}}
    save_json(output_dir/'document.json',result)
    save_json(output_dir/'sourceparts.json',parts)
    return result


