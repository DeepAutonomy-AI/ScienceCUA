from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urldefrag

from bs4 import BeautifulSoup, Tag

from .models import (
    EvidenceNode,
    EvidencePlan,
    EvidencePlanExclusion,
    EvidencePlanNode,
    EvidencePlanIssue,
    EvidencePlanValidationReport,
    EvidenceRevision,
    EvidenceSourceSpan,
    EvidenceSourceRef,
    EvidenceTree,
    SourcePart,
)
from .source_tools.workspace import SourceWorkspace


@dataclass(frozen=True, slots=True)
class EvidenceTreeArtifacts:
    tree: EvidenceTree
    plan_artifact_id: str
    validation_artifact_id: str
    tree_artifact_id: str


class EvidencePlanRejected(ValueError):
    def __init__(
        self,
        report: EvidencePlanValidationReport,
        validation_artifact_id: str | None = None,
    ) -> None:
        self.report = report
        self.validation_artifact_id = validation_artifact_id
        summary = "; ".join(issue.message for issue in report.issues[:5])
        super().__init__(summary or "EvidencePlan validation failed")


@dataclass(frozen=True, slots=True)
class _ResolvedSpan:
    source_part_id: str
    text_artifact_id: str
    start_offset: int
    end_offset: int
    content: str
    content_sha256: str
    locator: dict[str, Any]
    source_order: int


class EvidencePlanEngine:
    """Validate an Agent plan and materialize a source-grounded Evidence tree.

    The engine never accepts Agent-authored Evidence content. Every materialized
    character is copied from an immutable normalized SourcePart span.
    """

    def __init__(self, workspace: SourceWorkspace) -> None:
        self.workspace = workspace
        self.store = workspace.store

    def eligible_parts(self) -> list[SourcePart]:
        """Return text-bearing parser units that must be referenced or excluded.

        HTML keeps every parser semantic unit available to the architecture plan.
        For PDF, a full-page projection marked ``evidence_default=False`` is a
        fallback copy of text already represented by one or more exact section
        units. Keeping it would duplicate the same page in the candidate tree.
        """

        modality = self.workspace.manifest.primary_modality
        return [
            part
            for part in self.workspace.parts.values()
            if part.part_id in self.workspace.normalized_parts
            and part.text_artifact_id
            and self.workspace.part_text.get(part.part_id, "").strip()
            and not part.metadata.get("container")
            and not part.metadata.get("layout_atom")
            and not part.metadata.get("curated_projection")
            and not part.metadata.get("deterministic_projection")
            and not (
                modality == "pdf"
                and part.kind == "pdf_page_content"
                and part.metadata.get("evidence_default") is False
            )
        ]

    def build_skeleton_plan(self) -> EvidencePlan:
        """Derive a complete, shallow Evidence plan from parser-owned hierarchy.

        HTML uses the root page's authored visual/navigation groups and page graph.
        A fetched page is normally one Evidence node; authored fragment entries can
        reuse its unambiguous parsed subsection units. PDF uses parser-owned chapters/sections when present,
        otherwise page boundaries. No model judgment or generated prose participates.
        """

        eligible = self.eligible_parts()
        if not eligible:
            raise ValueError("Cannot build an Evidence skeleton without eligible SourceParts")
        if self.workspace.manifest.primary_modality == "html":
            nodes = self._build_html_skeleton_nodes(eligible)
        else:
            nodes = self._build_pdf_skeleton_nodes(eligible)
        return EvidencePlan(
            source_id=self.workspace.manifest.source_id,
            nodes=nodes,
            summary=(
                "Python-authored Evidence skeleton derived from parser-owned document "
                "hierarchy and source order."
            ),
        )

    def content_key(self, part: SourcePart) -> str:
        """Identify an existing unit without changing its frozen parser record."""

        if part.kind == "pdf_page_content" and part.metadata.get("content_key"):
            return str(part.metadata["content_key"])
        return self._stable_id("content", self.workspace.manifest.source_id, part.part_id)

    def content_registry(self) -> dict[str, str]:
        return {self.content_key(part): part.part_id for part in self.eligible_parts()}

    @staticmethod
    def _heading_label(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()

    def _matching_child_refs(
        self, parent: EvidencePlanNode, title: str,
    ) -> list[EvidenceSourceRef]:
        """Move only one unambiguous authored heading subtree from its parent.

        Parser ancestry, not prose search or page-specific markup, determines the
        subtree. Chunks share a section identity; repeated headings and multiple
        source pages require an explicit opaque content selection instead.
        """

        label = self._heading_label(title)
        if label == self._heading_label(parent.title):
            return []
        groups: dict[tuple[str, str, tuple[str, ...]], list[EvidenceSourceRef]] = {}
        for ref in parent.source_refs:
            part = self.workspace.parts.get(ref.source_part_id)
            visited: set[str] = set()
            while part is not None and part.part_id not in visited:
                visited.add(part.part_id)
                locator = part.locator
                heading = str(locator.get("heading") or "")
                if heading and self._heading_label(heading) == label:
                    path = locator.get("heading_path") or str(locator.get("section_path") or heading).split(" > ")
                    section_id = str(locator.get("structure_node_id") or locator.get("section_index") or part.part_id)
                    identity = (
                        str(locator.get("url") or self.workspace.manifest.source_id),
                        section_id,
                        tuple(self._heading_label(str(item)) for item in path),
                    )
                    groups.setdefault(identity, []).append(ref)
                    break
                parent_id = part.parent_part_id or locator.get("parent_section_part_id")
                part = self.workspace.parts.get(str(parent_id or ""))
        return next(iter(groups.values())) if len(groups) == 1 else []

    def apply_revision(
        self,
        base_plan: EvidencePlan,
        revision: EvidenceRevision,
    ) -> EvidencePlan:
        """Apply structural edits without accepting model-authored content."""

        if revision.source_id != self.workspace.manifest.source_id:
            raise ValueError("EvidenceRevision source_id does not match the current Source")
        nodes = [node.model_copy(deep=True) for node in base_plan.nodes]
        exclusions = [item.model_copy(deep=True) for item in base_plan.excluded_parts]

        def by_key() -> dict[str, EvidencePlanNode]:
            return {node.node_key: node for node in nodes}

        def descendants(node_key: str) -> list[str]:
            children: dict[str, list[str]] = {}
            for item in nodes:
                if item.parent_node_key:
                    children.setdefault(item.parent_node_key, []).append(item.node_key)
            found: list[str] = []

            def visit(key: str) -> None:
                for child in children.get(key, []):
                    found.append(child)
                    visit(child)

            visit(node_key)
            return found

        # Keys refer only to eligible units from this Source. Legacy PDF residual
        # keys remain valid; HTML/PDF section keys are derived without modifying
        # any parser-owned SourcePart, text, metadata, or source range.
        content_registry = self.content_registry()
        created_keys: set[str] = set()

        def content_part_ids(content_keys: list[str]) -> list[str]:
            missing = [key for key in content_keys if key not in content_registry]
            if missing:
                raise ValueError(f"Unknown or ineligible content_key values: {missing}")
            return [content_registry[key] for key in content_keys]

        def take_refs(part_ids: set[str]) -> list[EvidenceSourceRef]:
            moved: list[EvidenceSourceRef] = []
            for item in nodes:
                moved.extend(ref for ref in item.source_refs if ref.source_part_id in part_ids)
                item.source_refs = [ref for ref in item.source_refs if ref.source_part_id not in part_ids]
            if {ref.source_part_id for ref in moved} != part_ids:
                raise ValueError("content_key is not attached to the current Evidence skeleton")
            return moved

        def sort_refs(refs: list[EvidenceSourceRef]) -> None:
            order = {
                part.part_id: index for index, part in enumerate(self.eligible_parts())
            }
            refs.sort(
                key=lambda ref: (
                    order.get(ref.source_part_id, 10**9),
                    ref.start_offset or 0,
                )
            )

        for operation in revision.operations:
            current = by_key()
            node_key = str(operation.node_key or "")
            if operation.operation == "create_node":
                if node_key in current:
                    raise ValueError(
                        f"Evidence node_key already exists: {node_key!r}"
                    )
                parent_key = str(operation.parent_node_key or "")
                if parent_key not in current:
                    raise ValueError(
                        f"Unknown destination node_key: {parent_key!r}"
                    )
                created = EvidencePlanNode(
                    node_key=node_key,
                    title=str(operation.title or "").strip(),
                    title_origin="derived",
                    node_type=operation.node_type or "section",
                    parent_node_key=parent_key,
                    source_refs=[],
                )
                before_key = str(operation.before_node_key or "")
                if before_key:
                    if before_key not in current:
                        raise ValueError(f"Unknown before_node_key: {before_key!r}")
                    if current[before_key].parent_node_key != parent_key:
                        raise ValueError("before_node_key must be a child of parent_node_key")
                    nodes.insert(nodes.index(current[before_key]), created)
                else:
                    nodes.append(created)
                created_keys.add(node_key)
                if operation.content_keys:
                    created.source_refs = take_refs(set(content_part_ids(operation.content_keys)))
                    sort_refs(created.source_refs)
                else:
                    parent = current[parent_key]
                    created.source_refs = self._matching_child_refs(parent, created.title)
                    moved_ids = {id(ref) for ref in created.source_refs}
                    parent.source_refs = [ref for ref in parent.source_refs if id(ref) not in moved_ids]
                continue
            if operation.operation == "exclude_content":
                part_ids = set(content_part_ids(operation.content_keys))
                removed = take_refs(part_ids)
                exclusions.extend(
                    EvidencePlanExclusion(
                        source_part_id=ref.source_part_id,
                        start_offset=ref.start_offset,
                        end_offset=ref.end_offset,
                        reason=str(operation.reason),
                    )
                    for ref in removed
                )
                continue
            if node_key not in current:
                raise ValueError(f"Unknown Evidence skeleton node_key: {node_key!r}")
            target = current[node_key]
            if operation.operation == "assign_content":
                part_ids = set(content_part_ids(operation.content_keys))
                moved = take_refs(part_ids)
                target.source_refs.extend(moved)
                sort_refs(target.source_refs)
                continue
            if operation.operation == "update_node":
                if operation.title is not None:
                    target.title = operation.title.strip()
                    target.title_origin = "derived"
                if operation.node_type is not None:
                    target.node_type = operation.node_type
                continue
            if operation.operation == "move_node":
                parent_key = str(operation.parent_node_key or "")
                if target.parent_node_key is None:
                    raise ValueError("The Evidence root cannot be moved")
                if parent_key not in current:
                    raise ValueError(f"Unknown destination node_key: {parent_key!r}")
                if parent_key == node_key or parent_key in descendants(node_key):
                    raise ValueError("move_node would create a cycle")
                target.parent_node_key = parent_key
                before_key = str(operation.before_node_key or "")
                if before_key:
                    current = by_key()
                    if before_key not in current:
                        raise ValueError(f"Unknown before_node_key: {before_key!r}")
                    if before_key == node_key:
                        raise ValueError("A node cannot be positioned before itself")
                    if current[before_key].parent_node_key != parent_key:
                        raise ValueError("before_node_key must be a child of parent_node_key")
                    nodes.remove(target)
                    nodes.insert(nodes.index(current[before_key]), target)
                continue
            if operation.operation == "merge_nodes":
                source_keys = list(operation.source_node_keys)
                missing = [key for key in source_keys if key not in current]
                if missing:
                    raise ValueError(f"Unknown merge source node_key values: {missing}")
                if target.parent_node_key is None:
                    raise ValueError("The Evidence root cannot be a merge target")
                if any(current[key].parent_node_key != target.parent_node_key for key in source_keys):
                    raise ValueError("merge_nodes only accepts sibling source and target nodes")
                source_key_set = set(source_keys)
                merged_refs = list(target.source_refs)
                for key in source_keys:
                    merged_refs.extend(current[key].source_refs)
                sort_refs(merged_refs)
                target.source_refs = merged_refs
                for item in nodes:
                    if item.parent_node_key in source_key_set:
                        item.parent_node_key = target.node_key
                nodes[:] = [item for item in nodes if item.node_key not in source_key_set]
                continue
            if operation.operation == "exclude_subtree":
                if target.parent_node_key is None:
                    raise ValueError("The Evidence root cannot be excluded")
                removed_keys = {node_key, *descendants(node_key)}
                for item in nodes:
                    if item.node_key not in removed_keys:
                        continue
                    exclusions.extend(
                        EvidencePlanExclusion(
                            source_part_id=ref.source_part_id,
                            start_offset=ref.start_offset,
                            end_offset=ref.end_offset,
                            reason=str(operation.reason),
                        )
                        for ref in item.source_refs
                    )
                nodes[:] = [item for item in nodes if item.node_key not in removed_keys]

        # Removing a subtree can leave an empty structural wrapper. Prune those
        # wrappers, but never silently discard a newly requested node, including
        # one whose existing empty children are pruned in an earlier iteration.
        changed = True
        while changed:
            changed = False
            parents = {item.parent_node_key for item in nodes if item.parent_node_key}
            removable = {
                item.node_key
                for item in nodes
                if item.parent_node_key is not None
                and not item.source_refs
                and item.node_key not in parents
            }
            empty_created = sorted(removable & created_keys)
            if empty_created:
                raise ValueError(
                    f"New Evidence nodes have no unambiguous source content or children: {empty_created}. "
                    "Use content_keys returned by read_evidence_node (or PDF structure), "
                    "assign_content, or move existing nodes beneath a grouping parent."
                )
            if removable:
                nodes[:] = [item for item in nodes if item.node_key not in removable]
                changed = True

        return EvidencePlan(
            source_id=base_plan.source_id,
            nodes=nodes,
            excluded_parts=exclusions,
            summary=revision.summary or base_plan.summary,
        )

    def _build_html_skeleton_nodes(
        self, eligible: list[SourcePart]
    ) -> list[EvidencePlanNode]:
        pages = sorted(
            (part for part in self.workspace.parts.values() if part.kind == "html_page"),
            key=lambda item: (item.ordinal, item.part_id),
        )
        root_page = next(
            (part for part in pages if int(part.metadata.get("navigation_depth", 0)) == 0),
            pages[0] if pages else None,
        )
        root_url = str(root_page.locator.get("url") or "") if root_page else ""
        page_by_url = {
            str(page.locator.get("url")): page
            for page in pages
            if page.locator.get("url")
        }
        refs_by_url: dict[str, list[EvidenceSourceRef]] = {}
        unassigned: list[EvidenceSourceRef] = []
        for part in eligible:
            ref = EvidenceSourceRef(source_part_id=part.part_id)
            url = str(part.locator.get("url") or "")
            if url and url in page_by_url:
                refs_by_url.setdefault(url, []).append(ref)
            else:
                unassigned.append(ref)

        root_refs: list[EvidenceSourceRef] = list(unassigned)
        group_refs_by_label: dict[str, list[EvidenceSourceRef]] = {}
        for ref in refs_by_url.pop(root_url, []):
            part = self.workspace.parts.get(ref.source_part_id)
            group_label = (
                str(part.metadata.get("navigation_group_label") or "").strip()
                if part is not None
                else ""
            )
            if group_label:
                group_refs_by_label.setdefault(group_label.casefold(), []).append(ref)
            else:
                root_refs.append(ref)
        root_title = root_page.label if root_page is not None else self.workspace.manifest.display_name
        nodes: list[EvidencePlanNode] = [
            EvidencePlanNode(
                node_key="document",
                title=root_title,
                node_type="document",
                source_refs=root_refs,
            )
        ]

        groups = list(root_page.metadata.get("navigation_groups", [])) if root_page else []
        group_key_by_url: dict[str, str] = {}
        group_key_by_label: dict[str, str] = {}
        for index, group in enumerate(groups, start=1):
            key = f"topic_{index:03d}"
            authored_labels = [
                str(label).strip()
                for label in group.get("labels", [])
                if str(label).strip()
            ]
            if not authored_labels and str(group.get("label") or "").strip():
                authored_labels = [str(group["label"]).strip()]
            group_refs: list[EvidenceSourceRef] = []
            for label in authored_labels:
                group_refs.extend(group_refs_by_label.pop(label.casefold(), []))
            nodes.append(
                EvidencePlanNode(
                    node_key=key,
                    title=str(group.get("label") or f"Topic {index}")[:300],
                    node_type="section",
                    parent_node_key="document",
                    source_refs=group_refs,
                )
            )
            for url in group.get("links", []):
                group_key_by_url.setdefault(str(url), key)
            for label in group.get("labels", []):
                group_key_by_label[str(label).strip().casefold()] = key

        # A parsed disclosure catalog whose label could not be matched to a
        # visible navigation group remains grounded at the document root.
        for refs in group_refs_by_label.values():
            nodes[0].source_refs.extend(refs)

        page_key_by_url = {
            str(page.locator.get("url")): f"page_{index:03d}"
            for index, page in enumerate(pages, start=1)
            if page is not root_page and page.locator.get("url")
        }
        for page in pages:
            if page is root_page:
                continue
            url = str(page.locator.get("url") or "")
            parent_url = str(page.metadata.get("navigation_parent_url") or "")
            branch = str(page.metadata.get("capability_branch") or "").strip().casefold()
            parent_key = (
                page_key_by_url.get(parent_url)
                or group_key_by_url.get(url)
                or group_key_by_label.get(branch)
                or "document"
            )
            nodes.append(
                EvidencePlanNode(
                    node_key=page_key_by_url.get(url) or f"page_{len(nodes):03d}",
                    title=page.label,
                    node_type="section",
                    parent_node_key=parent_key,
                    source_refs=refs_by_url.pop(url, []),
                )
            )

        # Any parser unit whose page projection was unavailable remains grounded
        # directly at the document root rather than becoming uncovered.
        for refs in refs_by_url.values():
            nodes[0].source_refs.extend(refs)
        # A disclosure group without a visible directory match returns to the
        # root. Merge those references in authored order, including prose that
        # occurs after a disclosure, rather than appending an earlier span last.
        source_order = {part.part_id: index for index, part in enumerate(eligible)}
        for node in nodes:
            node.source_refs.sort(key=lambda ref: (source_order[ref.source_part_id], ref.start_offset or 0))
        nodes = self._apply_html_directory_topology(nodes, pages, root_page, group_key_by_label)
        if (self.workspace.settings.scope_selection_enabled
                or self.workspace.manifest.source_metadata.get("html_processing_mode") == "model_guided"):
            nodes = self._expand_html_heading_nodes(nodes)
        return self._prune_empty_skeleton_nodes(nodes)

    def _expand_html_heading_nodes(self, nodes: list[EvidencePlanNode]) -> list[EvidencePlanNode]:
        """Expose parsed page headings, including empty structural parents.

        Navigation establishes the page topology; SourcePart ancestry supplies
        page-local chapters without relying on guessed URL fragments.
        """
        result = []
        for owner in nodes:
            groups: dict[tuple[str, str], dict[str, Any]] = {}
            ref_group = {}
            for ref in owner.source_refs:
                part = self.workspace.parts[ref.source_part_id]
                chain, seen = [], set()
                while part and part.part_id not in seen and part.kind != "html_page":
                    seen.add(part.part_id)
                    locator = part.locator
                    heading = str(locator.get("heading") or "")
                    if (heading and int(locator.get("heading_level") or 0) > 0
                            and self._heading_label(heading) != self._heading_label(owner.title)):
                        identity = (str(locator.get("url") or ""), str(locator.get("section_index") or part.part_id))
                        chain.append((identity, heading))
                    parent = part.parent_part_id or locator.get("parent_section_part_id")
                    part = self.workspace.parts.get(str(parent or ""))
                parent_key = owner.node_key
                for identity, heading in reversed(chain):
                    if identity not in groups:
                        key = self._stable_id("heading", owner.node_key, *identity)
                        groups[identity] = {"node_key": key, "title": heading[:300],
                                            "parent_node_key": parent_key, "source_refs": []}
                    parent_key = groups[identity]["node_key"]
                if chain:
                    ref_group[ref.source_part_id] = chain[0][0]
            remaining = []
            for ref in owner.source_refs:
                identity = ref_group.get(ref.source_part_id)
                if identity is None:
                    remaining.append(ref)
                else:
                    groups[identity]["source_refs"].append(ref)
            owner.source_refs = remaining
            result.append(owner)
            result.extend(EvidencePlanNode(**payload) for payload in groups.values())
        return result

    def _apply_html_directory_topology(
        self,
        nodes: list[EvidencePlanNode],
        pages: list[SourcePart],
        root_page: SourcePart | None,
        group_key_by_label: dict[str, str],
    ) -> list[EvidencePlanNode]:
        """Use authored directory nesting/order, independently of crawl order.

        This only edits the plan's topology. Frozen SourceParts, their spans and
        attachments are unchanged; a missing parent page can supply a structural
        label, never invented content. Non-directory links cannot assign parents.
        """
        if root_page is None:
            return nodes
        artifact_id = root_page.metadata.get("root_structure_artifact_id")
        if not artifact_id:
            return nodes
        structure = self.store.read_json_artifact(str(artifact_id))
        by_key = {node.node_key: node for node in nodes}
        key_by_url = {
            str(page.locator.get("url")): (
                "document" if page is root_page else f"page_{index:03d}"
            )
            for index, page in enumerate(pages, start=1)
            if page.locator.get("url")
        }
        claimed = {"document"}
        ranks: dict[str, int] = {}
        page_by_url = {str(page.locator.get("url")): page for page in pages}
        fragment_documents: dict[str, BeautifulSoup] = {}
        parts_by_parent: dict[str, list[str]] = {}
        for part in self.workspace.parts.values():
            if part.parent_part_id:
                parts_by_parent.setdefault(part.parent_part_id, []).append(part.part_id)

        def rank(key: str) -> None:
            ranks.setdefault(key, len(ranks))

        def fragment_node(href: str, url: str, label: str, parent: str, owner: str) -> str | None:
            # Only an exact, unique parser heading may identify a subsection.
            # Ambiguous labels stay on the page; never guess or split its text.
            headings = [part for part in self.workspace.parts.values()
                        if part.locator.get("url") == url
                        and str(part.locator.get("heading") or "").strip().casefold() == label.casefold()
                        and int(part.locator.get("heading_level") or 0) > 0
                        and int(part.locator.get("chunk") or 1) == 1]
            if len(headings) != 1:
                return None
            page = page_by_url.get(url)
            if page is None or not page.artifact_id:
                return None
            if url not in fragment_documents:
                fragment_documents[url] = BeautifulSoup(
                    self.store.artifact_path(page.artifact_id).read_bytes(), "html.parser"
                )
            soup = fragment_documents[url]
            fragment = unquote(urldefrag(href)[1])
            targets = {id(tag): tag for tag in [
                *soup.find_all(id=fragment), *soup.find_all("a", attrs={"name": fragment})
            ]}
            if len(targets) != 1:
                return None
            target = next(iter(targets.values()))
            heading_tags = [f"h{level}" for level in range(1, 7)]
            heading = target if target.name in heading_tags else target.find_parent(heading_tags)
            if heading is None:
                heading = target.find(heading_tags) or target.find_next_sibling()
            if not isinstance(heading, Tag) or heading.name not in heading_tags:
                return None
            if " ".join(heading.get_text(" ", strip=True).split()).casefold() != " ".join(label.split()).casefold():
                return None
            key = self._stable_id("anchor", href)
            if key in by_key:
                return key
            owned: set[str] = set()
            pending = [headings[0].part_id]
            while pending:
                part_id = pending.pop()
                if part_id not in owned:
                    owned.add(part_id)
                    pending.extend(parts_by_parent.get(part_id, []))
            refs = [ref for ref in by_key[owner].source_refs if ref.source_part_id in owned]
            if not refs:
                return None
            by_key[owner].source_refs = [ref for ref in by_key[owner].source_refs if ref.source_part_id not in owned]
            by_key[key] = EvidencePlanNode(node_key=key, title=headings[0].label,
                parent_node_key=parent, node_type="section", source_refs=refs)
            nodes.append(by_key[key])
            rank(key)
            return key

        def visit(item: dict[str, Any], parent: str, path: str, authored: bool) -> None:
            children = item.get("children", [])
            href = str(item.get("href") or "")
            label = str(item.get("label") or "").strip()
            key = parent
            if item.get("in_scope") is not False and item.get("link_role") == "directory_entry" and href:
                url, fragment = urldefrag(href)
                target = key_by_url.get(url)
                if target is None and label:
                    target = self._stable_id("toc", url)
                    key_by_url[url] = target
                    node = EvidencePlanNode(node_key=target, title=label[:300],
                                            parent_node_key=parent, node_type="section")
                    nodes.append(node)
                    by_key[target] = node
                if target:
                    # A same-page fragment or repeated catalog link cannot make
                    # a page its own descendant or move a previously owned page.
                    ancestor = parent
                    while ancestor and ancestor != target:
                        ancestor = by_key[ancestor].parent_node_key
                    section_key = fragment_node(href, url, label, parent, target) if fragment and ancestor else None
                    if target not in claimed and not ancestor:
                        by_key[target].parent_node_key = parent
                        claimed.add(target)
                        rank(target)
                    key = section_key or target
            elif not href and label and children and authored:
                key = self._stable_id("toc", path, label)
                by_key[key] = EvidencePlanNode(node_key=key, title=label[:300],
                                               parent_node_key=parent, node_type="section")
                nodes.append(by_key[key])
                rank(key)
            # A prose reference is not a directory grouping, even if its markup
            # happens to contain a nested list.
            for index, child in enumerate(children):
                visit(child, key, f"{path}/{index}", authored)

        for index, tree in enumerate(structure.get("nested_link_trees", [])):
            context = str(tree.get("context_label") or "").strip().casefold()
            parent = group_key_by_label.get(context, "document")
            if by_key[parent].title.strip().casefold() == by_key["document"].title.strip().casefold():
                parent = "document"
            if parent != "document":
                rank(parent)
            for item_index, item in enumerate(tree.get("items", [])):
                visit(item, parent, f"{index}/{item_index}", bool(tree.get("authored_contents")))
        if not ranks:
            return nodes

        # Descendants first observed on child pages still follow those pages'
        # source link order. Sampling priority is never a presentation ordinal.
        for page in pages:
            for url in page.metadata.get("controlled_navigation", []):
                key = key_by_url.get(str(url))
                if key:
                    rank(key)
        original_order = {node.node_key: index for index, node in enumerate(nodes)}
        children_by_key: dict[str, list[EvidencePlanNode]] = {key: [] for key in by_key}
        for node in nodes:
            if node.parent_node_key:
                children_by_key[node.parent_node_key].append(node)
        ordered: list[EvidencePlanNode] = []

        def append_subtree(key: str) -> None:
            ordered.append(by_key[key])
            for child in sorted(children_by_key[key], key=lambda node: (
                ranks.get(node.node_key, len(ranks) + original_order[node.node_key]),
            )):
                append_subtree(child.node_key)

        append_subtree("document")
        return ordered

    def _build_pdf_skeleton_nodes(
        self, eligible: list[SourcePart]
    ) -> list[EvidencePlanNode]:
        """Build PDF topology from semantic navigation, never from page fallback units.

        ``pdf_section`` parts are content assignments to the parser-selected
        navigation tree.  Remaining eligible page projections stay grounded on
        the document root and are exposed separately to the PDF Child as
        unassigned content.  Consequently a page boundary can no longer become
        an Evidence node merely because its text was not mapped to a bookmark.
        """

        ordered = sorted(
            eligible,
            key=lambda part: (
                int(part.locator.get("page_start") or part.locator.get("page") or 0),
                int(part.ordinal),
                part.part_id,
            ),
        )
        section_parts = [part for part in ordered if part.kind == "pdf_section"]
        residual_parts = [part for part in ordered if part.kind != "pdf_section"]
        root = EvidencePlanNode(
            node_key="document",
            title=self.workspace.manifest.display_name,
            node_type="document",
            source_refs=[
                EvidenceSourceRef(source_part_id=part.part_id)
                for part in residual_parts
            ],
        )
        nodes = [root]
        key_by_part_id = {
            part.part_id: f"section_{index:03d}"
            for index, part in enumerate(section_parts, start=1)
        }
        for part in section_parts:
            parent_part_id = str(part.locator.get("parent_section_part_id") or "")
            parent_key = key_by_part_id.get(parent_part_id, "document")
            nodes.append(
                EvidencePlanNode(
                    node_key=key_by_part_id[part.part_id],
                    title=part.label[:300],
                    node_type="section",
                    parent_node_key=parent_key,
                    source_refs=[EvidenceSourceRef(source_part_id=part.part_id)],
                )
            )
        return nodes

    @staticmethod
    def _prune_empty_skeleton_nodes(
        nodes: list[EvidencePlanNode],
    ) -> list[EvidencePlanNode]:
        result = list(nodes)
        changed = True
        while changed:
            changed = False
            parents = {node.parent_node_key for node in result if node.parent_node_key}
            removable = {
                node.node_key
                for node in result
                if node.parent_node_key is not None
                and not node.source_refs
                and node.node_key not in parents
            }
            if removable:
                result = [node for node in result if node.node_key not in removable]
                changed = True
        return result

    def validate(self, plan: EvidencePlan) -> EvidencePlanValidationReport:
        report, _ = self._evaluate(plan)
        return report

    def materialize(self, plan: EvidencePlan) -> EvidenceTreeArtifacts:
        plan_artifact = self.store.put_json_artifact(
            plan.model_dump(mode="json"),
            kind="evidence_plan",
            source_id=self.workspace.manifest.source_id,
            metadata={"schema_version": plan.schema_version},
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(plan_artifact.artifact_id)
        report, resolved_by_node = self._evaluate(plan)
        validation_artifact = self.store.put_json_artifact(
            report.model_dump(mode="json"),
            kind="evidence_plan_validation",
            source_id=self.workspace.manifest.source_id,
            metadata={
                "schema_version": report.schema_version,
                "plan_artifact_id": plan_artifact.artifact_id,
                "valid": report.valid,
            },
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(validation_artifact.artifact_id)
        if not report.valid:
            raise EvidencePlanRejected(report, validation_artifact.artifact_id)

        node_ids = {
            node.node_key: self.store.canonical_id(
                "N",
                self._stable_id(
                    "evn", self.workspace.manifest.source_id,
                    plan_artifact.artifact_id, node.node_key,
                ),
                self.workspace.manifest.source_id,
                plan_artifact.artifact_id,
                node.node_key,
            )
            for node in plan.nodes
        }
        children_by_key: dict[str, list[str]] = {
            node.node_key: [] for node in plan.nodes
        }
        for node in plan.nodes:
            if node.parent_node_key:
                children_by_key[node.parent_node_key].append(node.node_key)

        node_payloads: dict[str, dict[str, Any]] = {}
        sibling_ordinals: dict[str | None, int] = {}
        for node in plan.nodes:
            parent_key = node.parent_node_key
            ordinal = sibling_ordinals.get(parent_key, 0)
            sibling_ordinals[parent_key] = ordinal + 1
            resolved = resolved_by_node.get(node.node_key, [])
            source_spans = [
                EvidenceSourceSpan(
                    source_part_id=span.source_part_id,
                    text_artifact_id=span.text_artifact_id,
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    content_sha256=span.content_sha256,
                    locator=span.locator,
                )
                for span in resolved
            ]
            content = "\n\n".join(span.content for span in resolved)
            content_artifact_id: str | None = None
            content_sha256: str | None = None
            if content:
                assembly_map: list[dict[str, Any]] = []
                cursor = 0
                for index, span in enumerate(resolved):
                    if index:
                        cursor += 2
                    start = cursor
                    cursor += len(span.content)
                    assembly_map.append(
                        {
                            "source_part_id": span.source_part_id,
                            "source_start_offset": span.start_offset,
                            "source_end_offset": span.end_offset,
                            "assembled_start_offset": start,
                            "assembled_end_offset": cursor,
                            "content_sha256": span.content_sha256,
                        }
                    )
                content_artifact = self.store.put_artifact(
                    content.encode("utf-8"),
                    kind="evidence_node_content",
                    media_type="text/plain",
                    suffix=".txt",
                    source_id=self.workspace.manifest.source_id,
                    metadata={
                        "node_key": node.node_key,
                        "plan_artifact_id": plan_artifact.artifact_id,
                        "source_spans": assembly_map,
                    },
                )
                content_artifact_id = content_artifact.artifact_id
                content_sha256 = content_artifact.sha256
                self.workspace.allowed_artifacts.add(content_artifact.artifact_id)
            node_payloads[node.node_key] = {
                "evidence_node_id": node_ids[node.node_key],
                "source_id": plan.source_id,
                "node_key": node.node_key,
                "parent_node_id": node_ids.get(parent_key) if parent_key else None,
                "child_node_ids": [node_ids[key] for key in children_by_key[node.node_key]],
                "ordinal": ordinal,
                "node_type": node.node_type,
                "title": node.title,
                "title_origin": node.title_origin,
                "source_refs": source_spans,
                "content_artifact_id": content_artifact_id,
                "content_sha256": content_sha256,
                "content": content,
                "direct_char_count": len(content),
                "descendant_char_count": 0,
            }

        def subtree_chars(node_key: str) -> int:
            descendant = sum(subtree_chars(child) for child in children_by_key[node_key])
            node_payloads[node_key]["descendant_char_count"] = descendant
            return int(node_payloads[node_key]["direct_char_count"]) + descendant

        root_key = next(node.node_key for node in plan.nodes if node.parent_node_key is None)
        subtree_chars(root_key)
        nodes = [EvidenceNode.model_validate(node_payloads[node.node_key]) for node in plan.nodes]
        identity = {
            "source_id": plan.source_id,
            "plan_artifact_id": plan_artifact.artifact_id,
            "node_ids": [node.evidence_node_id for node in nodes],
        }
        tree = EvidenceTree(
            evidence_tree_id=self.store.canonical_id(
                "ET",
                self._stable_id(
                    "evt", json.dumps(identity, sort_keys=True, separators=(",", ":")),
                ),
                json.dumps(identity, sort_keys=True, separators=(",", ":")),
            ),
            source_id=plan.source_id,
            root_node_id=node_ids[root_key],
            nodes=nodes,
            plan_artifact_id=plan_artifact.artifact_id,
            validation_artifact_id=validation_artifact.artifact_id,
            excluded_parts=plan.excluded_parts,
            referenced_part_count=report.referenced_part_count,
            leaf_count=sum(not node.child_node_ids for node in nodes),
        )
        tree_artifact = self.store.put_json_artifact(
            tree.model_dump(mode="json"),
            kind="evidence_tree",
            source_id=self.workspace.manifest.source_id,
            metadata={
                "schema_version": tree.schema_version,
                "evidence_tree_id": tree.evidence_tree_id,
                "plan_artifact_id": plan_artifact.artifact_id,
            },
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(tree_artifact.artifact_id)
        self.store.write_source_json(
            self.workspace.manifest.source_id,
            "evidence_tree.json",
            tree.model_dump(mode="json"),
        )
        self.store.emit(
            "evidence_tree_materialized",
            {
                "evidence_tree_id": tree.evidence_tree_id,
                "tree_artifact_id": tree_artifact.artifact_id,
                "node_count": len(nodes),
                "leaf_count": tree.leaf_count,
                "referenced_part_count": tree.referenced_part_count,
            },
            source_id=self.workspace.manifest.source_id,
            agent="evidence_plan_engine",
        )
        self.workspace.persist()
        return EvidenceTreeArtifacts(
            tree=tree,
            plan_artifact_id=plan_artifact.artifact_id,
            validation_artifact_id=validation_artifact.artifact_id,
            tree_artifact_id=tree_artifact.artifact_id,
        )

    def _evaluate(
        self, plan: EvidencePlan
    ) -> tuple[EvidencePlanValidationReport, dict[str, list[_ResolvedSpan]]]:
        issues: list[EvidencePlanIssue] = []
        eligible = self.eligible_parts()
        eligible_by_id = {part.part_id: part for part in eligible}
        source_order = {
            part_id: index for index, part_id in enumerate(self.workspace.parts)
        }
        resolved_by_node: dict[str, list[_ResolvedSpan]] = {}
        reference_intervals: dict[str, list[tuple[int, int, str]]] = {}
        exclusion_intervals: dict[str, list[tuple[int, int, str]]] = {}

        if plan.source_id != self.workspace.manifest.source_id:
            issues.append(
                EvidencePlanIssue(
                    code="source_scope_mismatch",
                    message="EvidencePlan source_id does not match the current Source.",
                )
            )

        for node in plan.nodes:
            resolved: list[_ResolvedSpan] = []
            submitted_order: list[tuple[int, int]] = []
            for ref in node.source_refs:
                span = self._resolve_span(
                    ref.source_part_id,
                    ref.start_offset,
                    ref.end_offset,
                    eligible_by_id,
                    source_order,
                    issues,
                    node_key=node.node_key,
                )
                if span is None:
                    continue
                resolved.append(span)
                submitted_order.append((span.source_order, span.start_offset))
                reference_intervals.setdefault(span.source_part_id, []).append(
                    (span.start_offset, span.end_offset, node.node_key)
                )
            if submitted_order != sorted(submitted_order):
                issues.append(
                    EvidencePlanIssue(
                        code="source_refs_out_of_order",
                        message="source_refs must preserve parser-owned source order.",
                        node_key=node.node_key,
                    )
                )
            resolved_by_node[node.node_key] = resolved

        for exclusion in plan.excluded_parts:
            span = self._resolve_span(
                exclusion.source_part_id,
                exclusion.start_offset,
                exclusion.end_offset,
                eligible_by_id,
                source_order,
                issues,
            )
            if span is not None:
                exclusion_intervals.setdefault(span.source_part_id, []).append(
                    (span.start_offset, span.end_offset, exclusion.reason)
                )

        for part_id, intervals in reference_intervals.items():
            self._report_overlaps(
                part_id,
                intervals,
                issues,
                code="overlapping_source_refs",
                message="The same source text is referenced by more than one Evidence node or range.",
            )
        for part_id, intervals in exclusion_intervals.items():
            self._report_overlaps(
                part_id,
                intervals,
                issues,
                code="overlapping_exclusions",
                message="Excluded ranges for one SourcePart overlap.",
            )
        for part_id in set(reference_intervals) & set(exclusion_intervals):
            for ref_start, ref_end, node_key in reference_intervals[part_id]:
                if any(
                    ref_start < excluded_end and excluded_start < ref_end
                    for excluded_start, excluded_end, _ in exclusion_intervals[part_id]
                ):
                    issues.append(
                        EvidencePlanIssue(
                            code="referenced_and_excluded_overlap",
                            message="A source range cannot be both referenced and excluded.",
                            node_key=node_key,
                            source_part_id=part_id,
                        )
                    )
                    break

        covered_parts: set[str] = set()
        for part in eligible:
            text = self.workspace.part_text[part.part_id]
            intervals = [
                (start, end)
                for start, end, _ in (
                    reference_intervals.get(part.part_id, [])
                    + exclusion_intervals.get(part.part_id, [])
                )
            ]
            if intervals:
                covered_parts.add(part.part_id)
            if self._has_uncovered_non_whitespace(text, intervals):
                issues.append(
                    EvidencePlanIssue(
                        code="uncovered_source_text",
                        message=(
                            "Every non-whitespace source span must be referenced or explicitly excluded."
                        ),
                        source_part_id=part.part_id,
                    )
                )

        referenced_parts = set(reference_intervals)
        excluded_parts = set(exclusion_intervals)
        report = EvidencePlanValidationReport(
            source_id=self.workspace.manifest.source_id,
            valid=not issues,
            issues=issues,
            eligible_part_count=len(eligible),
            referenced_part_count=len(referenced_parts),
            excluded_part_count=len(excluded_parts),
            coverage_complete=len(covered_parts) == len(eligible)
            and not any(issue.code == "uncovered_source_text" for issue in issues),
        )
        return report, resolved_by_node

    def _resolve_span(
        self,
        source_part_id: str,
        start_offset: int | None,
        end_offset: int | None,
        eligible_by_id: dict[str, SourcePart],
        source_order: dict[str, int],
        issues: list[EvidencePlanIssue],
        *,
        node_key: str | None = None,
    ) -> _ResolvedSpan | None:
        part = eligible_by_id.get(source_part_id)
        if part is None:
            code = (
                "source_part_not_referenceable"
                if source_part_id in self.workspace.parts
                else "unknown_source_part"
            )
            issues.append(
                EvidencePlanIssue(
                    code=code,
                    message="The referenced SourcePart is not an eligible text unit.",
                    node_key=node_key,
                    source_part_id=source_part_id,
                )
            )
            return None
        text = self.workspace.part_text[source_part_id]
        start = 0 if start_offset is None else int(start_offset)
        end = len(text) if end_offset is None else int(end_offset)
        if start < 0 or end > len(text) or end <= start:
            issues.append(
                EvidencePlanIssue(
                    code="source_span_out_of_bounds",
                    message=f"Source span must be within 0..{len(text)}.",
                    node_key=node_key,
                    source_part_id=source_part_id,
                )
            )
            return None
        content = text[start:end]
        if not content.strip():
            issues.append(
                EvidencePlanIssue(
                    code="empty_source_span",
                    message="A referenced or excluded source span cannot be whitespace only.",
                    node_key=node_key,
                    source_part_id=source_part_id,
                )
            )
            return None
        return _ResolvedSpan(
            source_part_id=source_part_id,
            text_artifact_id=str(part.text_artifact_id),
            start_offset=start,
            end_offset=end,
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            locator=dict(part.locator),
            source_order=source_order[source_part_id],
        )

    @staticmethod
    def _report_overlaps(
        part_id: str,
        intervals: list[tuple[int, int, str]],
        issues: list[EvidencePlanIssue],
        *,
        code: str,
        message: str,
    ) -> None:
        ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                issues.append(
                    EvidencePlanIssue(
                        code=code,
                        message=message,
                        source_part_id=part_id,
                    )
                )
                return

    @staticmethod
    def _has_uncovered_non_whitespace(
        text: str, intervals: list[tuple[int, int]]
    ) -> bool:
        if not intervals:
            return bool(text.strip())
        cursor = 0
        for start, end in sorted(intervals):
            if text[cursor:start].strip():
                return True
            cursor = max(cursor, end)
        return bool(text[cursor:].strip())

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return f"{prefix}_{digest[:24]}"
