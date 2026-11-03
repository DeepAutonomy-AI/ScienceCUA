"""Resolve HTML addresses and describe link roles without site-specific URL rules."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


HEADING_TAGS = {f"h{level}" for level in range(1, 7)}


def html_base_url(soup: BeautifulSoup, page_url: str) -> str:
    """Use the first base[href], resolved against the final document address."""
    base = soup.find("base", href=True)
    if isinstance(base, Tag):
        candidate = urljoin(page_url, str(base.get("href") or "").strip())
        if urlparse(candidate).scheme in {"http", "https", "file"}:
            return candidate
    return page_url


def _ancestors(anchor: Tag) -> list[Tag]:
    return [anchor, *[node for node in anchor.parents if isinstance(node, Tag)]]


def _tokens(node: Tag) -> set[str]:
    return {
        str(node.get("id") or "").casefold(),
        *(str(value).casefold() for value in node.get("class", [])),
    }


def is_heading_permalink(anchor: Tag) -> bool:
    title = str(anchor.get("title") or "").strip().casefold()
    if title in {"permalink", "permanent link", "link to this heading", "link to this section"}:
        return True
    for node in _ancestors(anchor):
        if any(re.search(r"(?:^|[-_])(permalink|headerlink|header-link)(?:$|[-_])", token)
               for token in _tokens(node)):
            return True
        if node.name in HEADING_TAGS:
            rel = {str(value).casefold() for value in anchor.get("rel", [])}
            return "bookmark" in rel
        if node.name in {"main", "article", "body"}:
            break
    return False


def is_authored_contents(anchor: Tag) -> bool:
    """Recognize explicit contents containers, including those implemented as nav."""
    for node in _ancestors(anchor):
        if str(node.get("role") or "").casefold() == "doc-toc":
            return True
        if any(re.search(r"(?:^|[-_])(toc|toctree|contents|directory)(?:$|[-_])", token)
               for token in _tokens(node)):
            return True
        if node.name == "nav" and re.search(
            r"\b(contents|table of contents)\b", str(node.get("aria-label") or ""), re.I
        ):
            return True
    return False


def _outside_link_text(container: Tag, *, exclude_nested_lists: bool = False) -> str:
    pieces: list[str] = []
    for value in container.descendants:
        if not isinstance(value, NavigableString):
            continue
        parents = []
        for parent in value.parents:
            if parent is container:
                break
            if isinstance(parent, Tag):
                parents.append(parent)
        if any(parent.name == "a" for parent in parents):
            continue
        if exclude_nested_lists and any(parent.name in {"ul", "ol"} for parent in parents):
            continue
        pieces.append(str(value))
    return " ".join(pieces).strip()


def _is_link_entry(container: Tag, *, exclude_nested_lists: bool = False) -> bool:
    # Chapter numbers, punctuation and icons may accompany a directory label.
    outside = _outside_link_text(container, exclude_nested_lists=exclude_nested_lists)
    return not any(character.isalpha() for character in outside)


def _is_described_entry(anchor: Tag, container: Tag) -> bool:
    """A leading linked term followed by a separator owns its catalog entry.

    The description may contain other links; those remain cross-references.
    Nested lists belong to their own entries, and prose before a link cannot
    establish a directory relationship.
    """
    list_tags = ["ul", "ol", "dl"]
    owner = container.find_parent(list_tags)
    first = next((node for node in container.find_all("a", href=True)
                  if node.find_parent(list_tags) is owner), None)
    if first is not anchor:
        return False
    before, after = [], []
    found = False
    for value in container.descendants:
        if not isinstance(value, NavigableString) or isinstance(value, Comment):
            continue
        if value.find_parent(list_tags) is not owner:
            continue
        if value.find_parent("a") is anchor:
            found = True
            continue
        (after if found else before).append(str(value))
    return (found and not any(character.isalpha() for character in "".join(before))
            and bool(re.match(r"^\s*(?:[–—:：]|-(?=\s))\s*", "".join(after))))


def classify_html_link(anchor: Tag) -> tuple[str, str]:
    """Return a role and its DOM evidence; ambiguous links never imply children."""
    if is_heading_permalink(anchor):
        return "heading_permalink", "permalink_markup"
    if str(anchor.get("href") or "").strip().startswith("#"):
        return "local_anchor", "fragment_only"
    rel = {str(value).casefold() for value in anchor.get("rel", [])}
    if rel & {"prev", "next"}:
        return "page_navigation", "prev_next_relation"
    ancestors = _ancestors(anchor)
    if any(node.name in {"script", "style", "noscript", "template", "header", "footer"}
           for node in ancestors):
        return "site_navigation", "outside_document_content"
    if anchor.find_parent(["pre", "code"]):
        return "cross_reference", "code_reference"
    if is_authored_contents(anchor):
        return "directory_entry", "authored_contents_container"
    entry = anchor.find_parent(["li", "dt", "td", "th"])
    if isinstance(entry, Tag) and _is_described_entry(anchor, entry):
        return "directory_entry", "linked_term_with_description"
    paragraph = anchor.find_parent("p")
    if isinstance(paragraph, Tag) and not _is_link_entry(paragraph):
        return "cross_reference", "inline_prose"
    item = anchor.find_parent("li")
    if isinstance(item, Tag):
        if _is_link_entry(item, exclude_nested_lists=True):
            return "directory_entry", "linked_list_entry"
        return "cross_reference", "inline_list_prose"
    # Older manuals use bare anchors after headings, including cards in table cells.
    parent = anchor.parent
    standalone = isinstance(parent, Tag) and not any(
        any(character.isalpha() for character in str(value))
        for value in parent.children if isinstance(value, NavigableString)
    )
    previous = anchor.find_previous_sibling()
    if paragraph is None and standalone and isinstance(previous, Tag) and previous.name in {*HEADING_TAGS, "a", "br"}:
        return "directory_entry", "standalone_catalog_entry"
    if anchor.find_parent(list(HEADING_TAGS)):
        return "directory_entry", "linked_heading"
    definition = anchor.find_parent("dt")
    if isinstance(definition, Tag) and _is_link_entry(definition):
        return "directory_entry", "linked_definition_entry"
    cell = anchor.find_parent(["td", "th"])
    if isinstance(cell, Tag):
        table = cell.find_parent("table")
        if _is_link_entry(cell) and isinstance(table, Tag) and len(table.find_all("a", href=True)) >= 2:
            return "directory_entry", "linked_table_entry"
        return "cross_reference", "table_reference"
    if isinstance(paragraph, Tag):
        # A lone linked paragraph needs an authored catalog context.
        details = paragraph.find_parent("details")
        if isinstance(details, Tag) and details.find("summary", recursive=False):
            return "directory_entry", "disclosure_entry"
        return "unclassified", "linked_paragraph_without_catalog"
    details = anchor.find_parent("details")
    if isinstance(details, Tag) and details.find("summary", recursive=False):
        return "directory_entry", "disclosure_entry"
    return "unclassified", "no_directory_evidence"
