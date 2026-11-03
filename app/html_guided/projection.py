"""Non-visible transport guards for faithful HTML5 projection round trips."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from bs4 import Comment, NavigableString, Tag

from ..html_sourcepart.extractor import parse_html


def _original_dom_path(node: Tag) -> str:
    pieces = []
    while isinstance(node, Tag) and node.name != "[document]":
        index = 1 + sum(1 for sibling in node.previous_siblings
                        if isinstance(sibling, Tag) and sibling.name == node.name)
        pieces.append(f"{node.name}:nth-of-type({index})")
        node = node.parent
    return " > ".join(reversed(pieces))


def preserve_initial_linefeeds(node: Tag) -> tuple[Tag, list[dict[str, Any]]]:
    """Copy a selected region and preserve its already-parsed leading LFs.

    HTML5 discards one LF immediately after the start of pre/textarea/listing.
    An original capture may retain another, authored LF after that first parse.
    Serializing it directly would discard that LF on the extractor's next parse.
    A preceding empty comment prevents the second discard in pre/listing.
    Textarea uses RCDATA, where a comment would become literal text: pad its
    serialized value with one LF instead. Apply this helper before EACH
    serialization of a newly parsed tree, including normalization passes.
    """
    cloned = copy.deepcopy(node)
    if node.name == "[document]":
        # BeautifulSoup's HTML5 document deepcopy constructs an empty HTML tree
        # before appending the copied original children. Rebuild those children
        # explicitly so the clone has exactly the original tree and tag order.
        # Tag deepcopy itself never reparses whitespace or adds a document.
        cloned.clear()
        for child in node.contents:
            cloned.append(copy.deepcopy(child))
    originals = [node, *node.find_all(True)]
    copies = [cloned, *cloned.find_all(True)]
    adjustments = []
    for original, copied in zip(originals, copies):
        if original.name not in {"pre", "textarea", "listing"} or not copied.contents:
            continue
        first = copied.contents[0]
        if (isinstance(first, NavigableString) and not isinstance(first, Comment)
                and str(first).startswith("\n")):
            path = _original_dom_path(original)
            if original.name == "textarea":
                first.replace_with(NavigableString("\n" + str(first)))
                kind = "textarea_initial_lf_padding"
                reason = "One transport-only linefeed preserves the captured textarea value after HTML5 consumes its initial linefeed"
            else:
                copied.insert(0, Comment(""))
                kind = "html5_initial_linefeed_guard"
                reason = "An empty non-visible comment preserves the captured initial linefeed on subsequent HTML5 parsing"
            adjustments.append({
                "kind": kind, "dom_path": path, "transport_only": True,
                "original_dom_path": path, "tag": original.name,
                "reason": reason,
            })
    return cloned, adjustments


def verify_projected_code(original_html: str, selected_dom_path: str,
                          projected_html: str) -> dict[str, Any]:
    """Check literal code against an independent parse of the raw capture.

    Hashing the projected document alone cannot detect an earlier parser's
    whitespace loss. Both sides are parsed independently here, using the same
    HTML5 rules as SourcePart extraction, before comparing ordered code values.
    """
    original = parse_html(original_html)
    if not selected_dom_path or selected_dom_path == "[document]":
        source_region = original
    else:
        try:
            source_region = original.select_one(selected_dom_path)
        except Exception:
            raise ValueError("The captured content boundary is invalid for code verification") from None
        if source_region is None:
            raise ValueError("The captured content boundary was not found for code verification")
    projected = parse_html(projected_html)
    projected_region = projected.find("main", attrs={"data-guided-projection": "true"})
    if projected_region is None:
        raise ValueError("The projected content boundary was not found for code verification")
    code_tags = {"pre", "code", "listing", "textarea"}

    def values(region: Tag) -> list[tuple[str, str]]:
        nodes = ([region] if region.name in code_tags else []) + region.find_all(list(code_tags))
        return [(node.name, node.get_text()) for node in nodes]

    source_values = values(source_region)
    projected_values = values(projected_region)
    if source_values != projected_values:
        raise ValueError(
            "HTML projection changed captured code or text-control values "
            f"(original nodes: {len(source_values)}, projected nodes: {len(projected_values)})"
        )

    def digest(value: list[tuple[str, str]]) -> str:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8", errors="surrogatepass")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "status": "passed", "code_nodes_checked": len(source_values),
        "original_code_sha256": digest(source_values),
        "projected_code_sha256": digest(projected_values),
    }
