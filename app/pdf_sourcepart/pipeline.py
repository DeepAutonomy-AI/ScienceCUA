"""Persist generic PDF structure in the application's immutable artifact store.

The parser owns readable blocks; the existing Evidence topology/review contract
continues to own section assignment. No experiment files are runtime inputs.
"""
from __future__ import annotations

import copy
import json
import mimetypes
import tempfile
from pathlib import Path

from ..models import SourcePart
from . import extractor
from .filesystem import filesystem_path


NORMALIZATION_VERSION = 'v15-pdf-structured.v4-visual-recovery'


def _json(workspace, value, kind):
    artifact = workspace.store.put_json_artifact(value, kind=kind,
        source_id=workspace.manifest.source_id, preserve_text_integrity=True)
    workspace.allowed_artifacts.add(artifact.artifact_id)
    return artifact.artifact_id


def _readable(block, links):
    if block['type'] == 'heading':
        return extractor.plain(block.get('runs', [])).strip()
    text = extractor.block_markdown(block)
    # Make the model-facing projection self-contained while retaining exact
    # link records separately (including destinations that are not clickable).
    for link in links:
        if link.get('clickable'):
            url = str(link['url']).replace(' ', '%20').replace('(', '%28').replace(')', '%29')
            text = text.replace(']['+link['link_id']+']', ']('+url+')')
    return text.strip()


def inventory_pages(executor, raw_id):
    from ..source_tools.pdf_structure import _classify_pdf_image
    from ..source_tools.text import _compact_text
    workspace, store = executor.workspace, executor.store
    source_id = workspace.manifest.source_id
    pages, layouts, contexts = [], [], []
    with tempfile.TemporaryDirectory(
        prefix='pdf-sourcepart-', dir=filesystem_path(store.root / 'incoming'),
    ) as temporary:
        directory = Path(temporary)
        document = extractor.extract_pdf(
            store.artifact_path(raw_id), directory, source_id=source_id,
            extraction_scope=workspace.manifest.source_metadata.get('extraction_scope'),
        )
        has_document_body = any(b['type'] in {'heading','paragraph','code','list','table','toc'}
                                for p in document['pages'] for b in p['blocks'])
        blank_document = not has_document_body and all(b['type']=='blank_page'
                              for p in document['pages'] for b in p['blocks'])
        if blank_document:
            workspace.gaps.append('PDF has no body text or visual content; explicit blank-page records and the original document are preserved.')
        resource_ids = {}
        # Persist each referenced binary before the staging directory disappears.
        filenames = {im[k] for im in document['images'] for k in ('filename','display_filename') if im.get(k)}
        filenames.update(b['filename'] for p in document['pages'] for b in p['blocks'] if b.get('filename'))
        filenames.update(Path(a['path']).name for a in document.get('assets', []))
        for filename in sorted(filenames):
            path = directory / 'assets' / filename
            media = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            artifact = store.put_artifact(path.read_bytes(), kind='pdf_embedded_image' if media.startswith('image/') else 'pdf_recovery_source',
                media_type=media, suffix=path.suffix, source_id=source_id)
            workspace.allowed_artifacts.add(artifact.artifact_id)
            resource_ids[filename] = artifact.artifact_id
        summaries = []
        for page in document['pages']:
            number = page['page']
            page_id = f'{source_id}_page_{number:05d}'
            raw_path = directory / 'raw' / f'page-{number:04d}.json'
            raw_snapshot = json.loads(raw_path.read_text(encoding='utf-8')) if raw_path.exists() else {}
            raw_text_path = directory / 'raw' / f'page-{number:04d}.txt'
            raw_text = (raw_text_path.read_text(encoding='utf-8') if raw_text_path.exists() else '') or page.get('text','')
            if page['mode']=='raw_pdf':
                raw_text = ''
            snapshot_id = _json(workspace, {'raw_artifact_id':raw_id, **raw_snapshot}, 'pdf_page_snapshot')
            saved_page = copy.deepcopy(page)
            for item in [*saved_page['blocks'], *saved_page.get('images', [])]:
                if item.get('filename'):
                    item['artifact_id'] = resource_ids[item['filename']]
                if item.get('display_filename'):
                    item['display_artifact_id'] = resource_ids[item['display_filename']]
            saved_page.update(raw_artifact_id=raw_id, source_snapshot_artifact_id=snapshot_id,
                              resource_artifacts=resource_ids)
            structure_id = _json(workspace, saved_page, 'pdf_page_structure')
            page_part = SourcePart(part_id=page_id, source_id=source_id, kind='pdf_page',
                label=f'Page {number}', locator={'page':number, 'page_number_known':page.get('page_number_known', True)},
                ordinal=number, artifact_id=raw_id,
                metadata={'container':True, 'width':page.get('width'), 'height':page.get('height'),
                          'extraction_mode':page['mode'], 'degraded':page['degraded'],
                          'structure_artifact_id':structure_id, 'source_snapshot_artifact_id':snapshot_id,
                          'resource_artifact_ids':list(resource_ids.values())})
            workspace.add_part(page_part);workspace.inventoried_parts.add(page_id)
            semantic_blocks, payloads = [], []
            for index, block in enumerate(page['blocks'], 1):
                if block['type'] in {'image','page_image'}:
                    continue
                text = _readable(block, page['links'])
                if not text:
                    continue
                block_id = f'{page_id}_block_{index:04d}'
                part = SourcePart(part_id=block_id, source_id=source_id, kind='pdf_text_block',
                    label=f'Page {number} {block["type"]} {index}', parent_part_id=page_id,
                    locator={'page':number,'bbox':block.get('bbox'),'line_ids':block.get('line_ids',[])},
                    ordinal=index, artifact_id=raw_id,
                    metadata={'layout_atom':True,'evidence_default':False,'structure_artifact_id':structure_id})
                workspace.add_part(part,text);workspace.inventoried_parts.add(block_id)
                sizes = [r.get('size',0) for r in block.get('runs',[]) if r.get('size')]
                semantic_blocks.append({'part_id':block_id,'page':number,'page_part_id':page_id,'text':text,
                    'bbox':block.get('bbox'),'ordinal':index, 'page_height':page.get('height') or 0,
                    'page_width':page.get('width') or 0, 'font_size':max(sizes,default=document.get('body_font_size') or 11),
                    'max_font_size':max(sizes,default=document.get('body_font_size') or 11),
                    'bold':block['type']=='heading', 'line_count':len(block.get('line_ids',[])) or 1,
                    'structured_block':saved_page['blocks'][index-1], 'structure_artifact_id':structure_id,
                    'structure_owned':not blank_document, 'heading_hint':block if block['type']=='heading' else None})
                payloads.append({'part_id':block_id,'bbox':block.get('bbox'),'char_count':len(text),
                                 'text_artifact_id':part.text_artifact_id,'type':block['type']})
            image_ids = []
            image_map = {im['image_id']:im for im in page.get('images',[])}
            # Display composites keep all constituent original resources in the
            # page structure. Evidence carries the coherent visual region.
            for index, block in enumerate(page['blocks'], 1):
                if block['type'] not in {'image','page_image'} or not block.get('filename'):
                    continue
                image = image_map.get(block.get('image_id'), {})
                asset = next((a for a in page.get('assets',[]) if a.get('image_id') == block.get('image_id')), {})
                width, height = int(image.get('width') or asset.get('width') or 0), int(image.get('height') or asset.get('height') or 0)
                rects = [block['bbox']] if block.get('bbox') else []
                role, reason, ratio = _classify_pdf_image(pixel_width=width,pixel_height=height,rects=rects,
                    page_width=float(page.get('width') or 0),page_height=float(page.get('height') or 0))
                composite = bool(block.get('rendered_composite'))
                # Small authored toolbar/button icons can carry instructions.
                # Pixel size alone is not evidence that such an icon is chrome.
                preserve_visual = composite or block['type']=='page_image' or (width > 4 and height > 4)
                if preserve_visual or not raw_text.strip():
                    role, reason = 'informative', 'structured_visual_preservation'
                image_id = f'{page_id}_image_{index:04d}'
                image_part = SourcePart(part_id=image_id,source_id=source_id,
                    kind='pdf_visual_group' if composite else 'pdf_image_region',label=f'Page {number} image {index}',
                    locator={'page':number,'bboxes':rects},parent_part_id=page_id,ordinal=index,
                    artifact_id=resource_ids[block['filename']], metadata={'preserved_binary':True,
                        'media_type':mimetypes.guess_type(block['filename'])[0] or 'image/png',
                        'pixel_width':width,'pixel_height':height,'role':role,'classification_reason':reason,
                        'display_area_ratio':ratio,'source_image_ids':block.get('source_image_ids',[]),
                        'structure_artifact_id':structure_id,'parser_preserved_visual':preserve_visual})
                workspace.add_part(image_part);workspace.inventoried_parts.add(image_id);image_ids.append(image_id)
                if role=='decorative':
                    workspace.skipped[image_id]='decorative_image_filtered'
            # Visual-only pages must have a text transport anchor for the native
            # EvidencePlan span contract. It is explicitly not OCR text.
            if not semantic_blocks and image_ids:
                text = '[Visual PDF page preserved; machine-readable body text is unavailable. OCR was not used.]'
                block_id = f'{page_id}_visual_anchor'
                semantic_blocks.append({'part_id':block_id,'page':number,'page_part_id':page_id,'text':text,
                    'bbox':None,'ordinal':1,'page_height':page.get('height') or 0,'page_width':page.get('width') or 0,
                    'font_size':11,'max_font_size':11,'line_count':1,'synthetic_visual_anchor':True,
                    'structured_block':{'type':'visual_anchor','text':text,'page':number},'structure_artifact_id':structure_id})
                if not has_document_body:
                    workspace.gaps.append(f'PDF page {number} has no machine-readable body; its visual content is retained without OCR.')
            text = '\n\n'.join(b['text'] for b in semantic_blocks)
            content_id = f'{page_id}_content'
            content_part = SourcePart(part_id=content_id,source_id=source_id,kind='pdf_page_content',label=f'Page {number}',
                locator={'page':number,'source_block_ids':[b['part_id'] for b in semantic_blocks]}, parent_part_id=page_id,
                ordinal=number,artifact_id=raw_id,metadata={'semantic_unit':True,'semantic_granularity':'page',
                    'evidence_default':True,'extraction_mode':page['mode'],'degraded':page['degraded'],
                    'structure_artifact_id':structure_id,'source_snapshot_artifact_id':snapshot_id,'links':page['links']})
            workspace.add_part(content_part,text);workspace.inventoried_parts.add(content_id)
            markdown = store.put_artifact(text.encode('utf-8'),kind='pdf_markdown_projection',media_type='text/markdown',suffix='.md',source_id=source_id)
            workspace.allowed_artifacts.add(markdown.artifact_id)
            page_part.metadata.update(markdown_artifact_id=markdown.artifact_id,page_content_part_id=content_id,
                text_block_count=len(payloads),image_count=len(page.get('images',[])))
            pages.append({'page':number,'part_id':page_id,'text_chars':len(text),'text_block_count':len(payloads),
                'image_count':len(image_ids),'image_part_ids':image_ids,'markdown_artifact_id':markdown.artifact_id,
                'page_content_part_id':content_id,'text_preview':_compact_text(text)[:180]})
            layouts.append({'page':number,'page_part_id':page_id,'blocks':payloads,'image_part_ids':image_ids,
                            'structure_artifact_id':structure_id,'source_snapshot_artifact_id':snapshot_id})
            contexts.append({'page':number,'page_part_id':page_id,'page_content_part_id':content_id,
                'blocks':semantic_blocks,'page_height':page.get('height') or 0,'image_part_ids':image_ids,
                'raw_text':raw_text,'extraction_mode':page['mode'],'degraded':page['degraded'],
                'links':page['links'],'structure_artifact_id':structure_id,
                'printed_toc_entries':[entry for entry in document['printed_toc'] if entry['toc_page']==number]})
            summaries.append({k:v for k,v in page.items() if k not in {'blocks','images','assets','annotations'}})
            if page['degraded']:
                workspace.gaps.append(f'PDF page {number} used {page["mode"]} recovery; original PDF and diagnostics are preserved in {structure_id}.')
            if page.get('warnings'):
                workspace.warnings.append(f'PDF page {number} extraction diagnostics are preserved in {structure_id}.')
        summary = {k:document[k] for k in ('schema_version','summary','opening_error','outline','printed_toc','body_font_size')}
        summary.update(raw_artifact_id=raw_id,pages=summaries,resource_artifact_ids=list(resource_ids.values()),
                       warnings=document.get('warnings',[]))
        artifact_id = _json(workspace,summary,'pdf_structured_source')
        store.emit('pdf_sourceparts_extracted', {'extractor_version':extractor.VERSION,'structured_source_artifact_id':artifact_id,
                   **document['summary']},source_id=source_id)
        return pages, layouts, contexts, document['outline'], summary


def attach_semantic_structure(workspace, contexts):
    """Keep exact readable offsets aligned with the section Evidence projection."""
    blocks = {b['part_id']:b for c in contexts for b in c['blocks']}
    for part in workspace.parts.values():
        if part.kind not in {'pdf_section','pdf_page_content'}:
            continue
        selected = [blocks[k] for k in part.locator.get('source_block_ids',[]) if k in blocks]
        content = workspace.part_text.get(part.part_id,'')
        ranges, cursor = [], 0
        for block in selected:
            position = content.find(block['text'].strip(),cursor)
            if position < 0:
                continue
            end = position + len(block['text'].strip())
            ranges.append({'start':position,'end':end,'block':block['structured_block']})
            cursor = end
        relevant = [c for c in contexts if c['page'] in {b['page'] for b in selected}]
        modes = sorted({c['extraction_mode'] for c in relevant})
        structure_id = _json(workspace, {'schema_version':NORMALIZATION_VERSION,'source_part_id':part.part_id,
            'blocks':[b['structured_block'] for b in selected],
            'block_ranges':ranges,'page_structure_artifact_ids':[c['structure_artifact_id'] for c in relevant],
            'links':[l for c in relevant for l in c['links']]},'pdf_sourcepart_structure')
        part.metadata.update(structure_artifact_id=structure_id,extraction_modes=modes,
            page_structure_artifact_ids=[c['structure_artifact_id'] for c in relevant],
            extraction_mode=modes[0] if len(modes)==1 else 'mixed',degraded=any(c['degraded'] for c in relevant),
            content_kinds=sorted({b['structured_block']['type'] for b in selected}))
        part.locator['structure_artifact_id']=structure_id
