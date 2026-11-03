"""Retrieved links retained independently of the bounded audit text preview."""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def retrieved_search_sources(output: str) -> list[dict[str, str]]:
    sources: list[Any] = []
    for section in output.split('DISCOVERY SEARCH EVIDENCE\n')[1:]:
        try:
            evidence, _ = json.JSONDecoder().raw_decode(section)
        except ValueError:
            continue
        if isinstance(evidence, dict) and isinstance(evidence.get('sources'), list):
            sources.extend(evidence['sources'])
    sources.extend({'title': title, 'url': url} for title, url in
        re.findall(r'^\s*-\s*\[([^\]]*)\]\((https?://[^\s]+)\)\s*$', output, re.MULTILINE))
    unique = {}
    for item in sources:
        if not isinstance(item, dict):
            continue
        try:
            url = str(item.get('url') or '').strip()
            parsed = urlsplit(url)
            if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or
                    parsed.username is not None or parsed.password is not None or
                    any(char.isspace() for char in url)):
                continue
            parsed.port
        except ValueError:
            continue
        if len(url) > 4000:
            continue
        key = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or '/', parsed.query, ''))
        unique.setdefault(key, {'url': url,
            'title': str(item.get('title') or '')[:500],
            'snippet': str(item.get('snippet') or '')[:1200]})
    # Keep PDF links ahead of unrelated results when applying the event bound.
    return sorted(unique.values(), key=lambda item: '.pdf' not in item['url'].casefold())[:40]
