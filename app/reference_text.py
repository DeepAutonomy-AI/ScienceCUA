"""Text-only Reference views; captured Evidence and saved source spans stay intact."""
from __future__ import annotations

import re
from typing import Any

from .models import ReferenceCandidate


def reference_presentation_content(reference: ReferenceCandidate | dict[str, Any]) -> str:
    """Read an accepted formatting view without modifying source or code text."""
    presentation = (reference.get('presentation') if isinstance(reference, dict)
                    else getattr(reference, 'presentation', None))
    if presentation is None:
        return ''
    mode = (presentation.get('mode') if isinstance(presentation, dict)
            else getattr(presentation, 'mode', None))
    content = (presentation.get('content') if isinstance(presentation, dict)
               else getattr(presentation, 'content', ''))
    return content if mode == 'agent_formatted' and isinstance(content, str) and content.strip() else ''


_IMAGE_MARKER = re.compile(r"\[image\s+[^\]\n]+\]", re.IGNORECASE)
_INLINE_IMAGE = re.compile(r"!\[[^\]\n]*\]\((?:\\.|[^()\\]|\([^()]*\))*\)")
_LINKED_IMAGE = re.compile(r"!\[([^\]\n]*)\]\[([^\]\n]*)\]")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def reference_text(value: str) -> str:
    """Remove image syntax while preserving every non-image text span."""
    value = str(value or '')
    image_labels = {match[2] or match[1] for match in _LINKED_IMAGE.finditer(value)}
    value = _LINKED_IMAGE.sub('', value)
    for label in image_labels:
        value = re.sub(r'(?im)^ {0,3}\[' + re.escape(label) + r'\]:[^\n]*(?:\n|$)', '', value)
    return _IMAGE_MARKER.sub('', _HTML_IMAGE.sub('', _INLINE_IMAGE.sub('', value)))


def reference_text_payload(reference: dict[str, Any]) -> dict[str, Any]:
    """Project only Reference fields, never mutate a corpus/Evidence container."""
    result = {key: value for key, value in reference.items() if key != 'image_refs'}
    for key in ('title', 'description', 'content', 'display_title', 'quote'):
        if isinstance(result.get(key), str):
            result[key] = reference_text(result[key])
    if 'citations' in result:
        result['citations'] = [reference_text_payload({key: value for key, value in citation.items()
            if key != 'image_artifact_id'}) for citation in result['citations']]
    if 'entries' in result:
        entries = []
        for original in result['entries']:
            entry = reference_text_payload(original)
            if not entry.get('title', '').strip():
                entry['title'] = entry.get('display_title', '').strip() or next((cell.strip() for cell in
                    re.split(r'(?<!\\)\|', entry.get('content', '').strip().strip('|')) if cell.strip()), '')
            entries.append(entry)
        result['entries'] = entries
    return result


def text_only_reference(reference: ReferenceCandidate) -> ReferenceCandidate:
    """Use a copy for rendering/export; source-exact validation uses originals."""
    payload = reference_text_payload(reference.model_dump(mode='json'))
    # Keep the typed presentation and its exact code/Markdown text. The generic
    # text-only projection is only for historical source fields and image refs.
    if hasattr(reference, 'presentation'):
        payload['presentation'] = reference.presentation
    entries = [entry.model_copy(update={**view, 'image_refs': [],
        'citations': [citation.model_copy(update={**raw, 'image_artifact_id': None})
            for citation, raw in zip(entry.citations, view['citations'], strict=True)]})
        for entry, view in zip(reference.entries, payload.get('entries', []), strict=True)]
    return reference.model_copy(update={**payload, 'entries': entries, 'image_refs': [],
        'citations': [citation.model_copy(update={**raw, 'image_artifact_id': None})
            for citation, raw in zip(reference.citations, payload['citations'], strict=True)]})
