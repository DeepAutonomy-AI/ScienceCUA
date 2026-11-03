"""Factual body/navigation boundary checks, independent of publisher selectors."""
from __future__ import annotations

from bs4 import Comment, NavigableString, Tag


_IGNORED = {"script", "style", "noscript", "template", "head", "title"}
_CHROME_TAGS = {"nav", "aside", "header", "footer"}
_CHROME_ROLES = {"navigation", "complementary", "banner", "contentinfo"}
_HEADINGS = {f"h{index}" for index in range(1, 7)}


def _roles(node: Tag) -> set[str]:
    return set(str(node.get("role") or "").casefold().split())


def _visible_text(node: Tag):
    for text in node.find_all(string=True):
        if isinstance(text, Comment) or not isinstance(text, NavigableString) or not str(text).strip():
            continue
        ancestors = [parent for parent in text.parents if isinstance(parent, Tag)]
        if any(parent.name in _IGNORED or parent.has_attr("hidden")
               or str(parent.get("aria-hidden") or "").casefold() == "true"
               for parent in ancestors):
            continue
        yield text, ancestors


def container_boundary_facts(node: Tag) -> dict:
    """Describe observed landmarks and text; never infer from IDs or classes."""
    navigation_text, body_text = 0, 0
    body_heading = body_prose = False
    for text, ancestors in _visible_text(node):
        if any(parent.name in _CHROME_TAGS or _roles(parent) & _CHROME_ROLES
               for parent in ancestors):
            navigation_text += 1
            continue
        body_text += 1
        # Limit authored-block checks to this candidate, not an outside parent.
        local = []
        for parent in ancestors:
            local.append(parent)
            if parent is node:
                break
        in_heading = any(parent.name in _HEADINGS for parent in local)
        body_heading = body_heading or in_heading
        # Generated manuals often use ordinary div/span blocks as paragraphs.
        # Do not require a publisher class or mistake a heading/link-only TOC
        # for prose. Text in explicit p/pre/code/table blocks also qualifies.
        explicit_prose = (not in_heading and
                          any(parent.name in {"p", "pre", "code", "table"} for parent in local))
        plain_prose = (not in_heading and not any(parent.name in {"a", "button", "label"} for parent in local)
                       and any(character.isalnum() for character in str(text))
                       and any(parent.name in {"div", "section", "span", "li", "dd", "dt", "td", "th"}
                               for parent in local))
        body_prose = body_prose or explicit_prose or plain_prose
    semantics = ("main" if node.name == "main" or "main" in _roles(node)
                 else "article" if node.name == "article" or "article" in _roles(node) else None)
    return {"role": str(node.get("role") or ""),
            "navigation_only": bool(navigation_text) and not body_text,
            "body_semantics": semantics,
            "authored_body": bool(body_heading and body_prose)}


def boundary_selection_error(selected_root_id: str, roots: dict[str, Tag]) -> str | None:
    """Ask for repair when navigation alone would omit a disjoint authored body.

    This does not choose a replacement. A parent retaining both the body and an
    authored sidebar is valid, as is navigation-only content without a competing
    explicit authored body. Link acquisition remains independent of this check.
    """
    selected = roots.get(selected_root_id)
    if selected is None or not container_boundary_facts(selected)["navigation_only"]:
        return None
    has_authored_alternative = False
    for identity, node in roots.items():
        if node is selected or selected in node.parents or node in selected.parents:
            continue
        facts = container_boundary_facts(node)
        if facts["body_semantics"] and facts["authored_body"] and not facts["navigation_only"]:
            has_authored_alternative = True
            break
    if not has_authored_alternative:
        return None
    return ("Selected content_root_id contains navigation text only and omits a separate explicit "
            "authored main/article body. "
            "Select a supplied region retaining the current page's complete authored body, or a "
            "parent retaining both body and owned catalog. Sidebar links remain available for "
            "acquisition decisions regardless of the chosen body boundary.")
