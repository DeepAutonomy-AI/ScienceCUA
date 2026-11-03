"""Preserve image anchors without treating page-local image numbers as global IDs."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .models import EvidenceExcerpt, ReferenceCandidate, ReferenceImageRef

IMAGE_MARKER = re.compile(r"\[image\s+[^\]\n]+\]")


def _payload(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _evidence_images(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    text = evidence.get("content", "")
    markers = list(IMAGE_MARKER.finditer(text))
    result = []
    for image in evidence.get("image_refs", []):
        # Anchors are SourcePart offsets, while marker matches are Evidence-local.
        offset = image.get("anchor_offset", 0) - evidence.get("start_offset", 0)
        match = next((m for m in markers if m.start() <= offset < m.end()), None)
        result.append({"artifact_id": image["artifact_id"],
                       "evidence_id": evidence["evidence_id"],
                       "media_type": image["media_type"],
                       "marker": match.group() if match else ""})
    return result


def _images_for_text(text: str, citations: list[dict[str, Any]],
                     evidence_by_id: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for citation in citations:
        evidence = evidence_by_id.get(citation.get("evidence_id"))
        if evidence is None:
            continue
        evidence = _payload(evidence)
        span_start = evidence.get("content", "").find(text) if text else -1
        for source_image, image in zip(evidence.get("image_refs", []), _evidence_images(evidence), strict=True):
            offset = source_image.get("anchor_offset", 0) - evidence.get("start_offset", 0)
            in_exact_span = not image["marker"] and span_start >= 0 and span_start <= offset < span_start + len(text)
            if ((image["marker"] and image["marker"] in text)
                    or in_exact_span
                    or image["artifact_id"] == citation.get("image_artifact_id")):
                if image not in result:
                    result.append(image)
    # A summary can cite several pages reusing I1. Never guess which page it means.
    ambiguous = {image["marker"] for image in result if image["marker"] and
                 len({r["artifact_id"] for r in result if r["marker"] == image["marker"]}) > 1}
    for image in result:
        if image["marker"] in ambiguous:
            image["marker"] = ""
    return result


def _display_title(entry: dict[str, Any]) -> str:
    title = entry.get("title", "")
    if not IMAGE_MARKER.search(title) or IMAGE_MARKER.sub("", title).strip():
        return title
    # Image-keyed table entries have no textual name. Use their exact source
    # description as a reading label, keeping title/content/locator untouched.
    text = IMAGE_MARKER.sub("", entry.get("content", ""))
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", text.strip().strip("|"))]
    return next((cell for cell in cells if cell), title)


def reference_image_payload(reference: dict[str, Any],
                            evidence_by_id: Mapping[str, Any]) -> dict[str, Any]:
    """Enrich new or historical References without changing source spans or citations."""
    result = dict(reference)
    entries = []
    for original in reference.get("entries", []):
        entry = dict(original)
        evidence_id = entry.get("source_evidence_id")
        citations = [{"evidence_id": evidence_id}, *entry.get("citations", [])]
        if evidence_id in evidence_by_id:
            entry["image_refs"] = _images_for_text(entry.get("content", ""), citations, evidence_by_id)
        entry["display_title"] = _display_title(entry)
        entries.append(entry)
    if "entries" in reference:
        result["entries"] = entries
    if any(c.get("evidence_id") in evidence_by_id for c in reference.get("citations", [])):
        result["image_refs"] = _images_for_text(reference.get("content", ""),
                                                  reference.get("citations", []), evidence_by_id)
    return result


def hydrate_reference_images(reference: ReferenceCandidate,
                             evidence_by_id: Mapping[str, EvidenceExcerpt]) -> ReferenceCandidate:
    payload = reference_image_payload(reference.model_dump(mode="json"), evidence_by_id)
    entries = [entry.model_copy(update={
        "image_refs": [ReferenceImageRef.model_validate(image) for image in enriched.get("image_refs", [])],
        "display_title": enriched["display_title"],
    }) for entry, enriched in zip(reference.entries, payload.get("entries", []), strict=True)]
    return reference.model_copy(update={"entries": entries, "image_refs": [
        ReferenceImageRef.model_validate(image) for image in payload.get("image_refs", [])]})
