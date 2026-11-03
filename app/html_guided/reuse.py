"""Exact-context reuse of model decisions within one acquisition scope.

This never decides whether a URL belongs to a manual. It only reuses an
explicit model decision for the same observed navigation, outside page bodies.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def navigation_signature(link: dict[str, Any], containers: list[dict[str, Any]]) -> str | None:
    occurrences = link.get('occurrences') or []
    if (not occurrences or link.get('navigation_source') or link.get('external_navigation_entries')
            or any(not item.get('in_chrome') or item.get('authored_contents') or item.get('in_code') for item in occurrences)):
        return None
    paths = {item['container_id']: item['dom_path'] for item in containers}
    facts = []
    for occurrence in occurrences:
        roots = occurrence.get('content_root_ids', [])
        if any(identity not in paths for identity in roots):
            return None
        facts.append({**{field: occurrence.get(field) for field in (
            'label', 'context', 'heading_path', 'disclosure_path', 'dom_path', 'in_chrome',
            'landmark_context', 'authored_contents', 'in_code', 'page_navigation', 'shared_navigation')},
            'container_paths': [paths[identity] for identity in roots]})
    payload = {field: link.get(field) for field in (
        'url', 'href', 'fragment', 'kind', 'fetchable', 'legacy_role', 'occurrence_count',
        'page_navigation_only', 'non_navigation_occurrence_count')}
    payload['occurrences'] = facts
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def outside_content(link: dict[str, Any], content_root_id: str) -> bool:
    occurrences = link.get('occurrences') or []
    return bool(occurrences) and all(content_root_id not in item.get('content_root_ids', [])
                                     for item in occurrences)
