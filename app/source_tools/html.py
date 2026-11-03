"""HTML normalization, bounded navigation, authored structure, and link auditing."""
from __future__ import annotations

import posixpath
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, NavigableString, Tag

from ..html_navigation import (
    classify_html_link,
    html_base_url,
    is_authored_contents,
    is_heading_permalink,
)
from ..html_sourcepart.fallback import Limits as HtmlLimits, scan as scan_html_budget
from ..html_sourcepart.pipeline import extract_page, semantic_nodes
from ..models import SourcePart
from .text import _compact_text, _stable_id


HTML_SECTION_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})
HTML_INLINE_HEADING_TAGS = frozenset({"h5", "h6"})
HTML_HEADING_TAGS = HTML_SECTION_HEADING_TAGS | HTML_INLINE_HEADING_TAGS


class HtmlToolsMixin:
    """HTML methods used by the Source-scoped ToolExecutor.

    The executor supplies workspace, store, settings, and common review helpers.
    """

    def _materialize_html_page(
        self,
        *,
        page_url: str,
        data: bytes,
        artifact_id: str,
        ordinal: int,
        navigation_depth: int,
        capability_branch: str | None,
        parent_url: str | None,
        authored_navigation_groups: list[str] | None = None,
        scope_targets: list[dict[str, Any]] | None = None,
    ) -> tuple[BeautifulSoup, SourcePart, dict[str, Any], list[tuple[SourcePart, str]]]:
        html = self._decode_html(data)
        _probe, budget_reasons = scan_html_budget(html, page_url, HtmlLimits())
        navigation_limited = any("budget" in item["code"] for item in budget_reasons)
        soup = BeautifulSoup("" if navigation_limited else html, "html.parser")
        if navigation_limited:
            recovery_note = (
                "selected bounds could not be resolved; the original capture is retained for audit only."
                if scope_targets is not None else
                "the complete captured HTML is still normalized through SourcePart recovery."
            )
            self.workspace.gaps.append(
                f"HTML navigation parsing was skipped for {page_url} because of complexity budgets; {recovery_note}"
            )
        title = _compact_text(soup.title.get_text(" ", strip=True) if soup.title else "")
        if not title:
            title = Path(page_url).name or self.workspace.manifest.display_name
        page_part_id = _stable_id("page", self.workspace.manifest.source_id, page_url)
        metadata: dict[str, Any] = {
            "container": True,
            "navigation_depth": navigation_depth,
        }
        if capability_branch:
            metadata["capability_branch"] = capability_branch
        if authored_navigation_groups:
            metadata["authored_navigation_groups"] = list(
                dict.fromkeys(authored_navigation_groups)
            )
        if parent_url:
            metadata["navigation_parent_url"] = parent_url
        page_part = SourcePart(
            part_id=page_part_id,
            source_id=self.workspace.manifest.source_id,
            kind="html_page",
            label=title[:300],
            locator={"url": page_url},
            ordinal=ordinal,
            artifact_id=artifact_id,
            metadata=metadata,
        )
        self.workspace.add_part(page_part)
        self.workspace.inventoried_parts.add(page_part_id)
        resource_base = (
            self.workspace.manifest.acquisition.get("local_resource_base_url")
            if navigation_depth == 0 else None
        )
        _document, sections, extraction_info = extract_page(
            self.workspace, html=html, soup=soup, page_url=page_url,
            raw_artifact_id=artifact_id, page_part_id=page_part_id,
            page_ordinal=ordinal, resource_base_url=resource_base,
            scope_targets=scope_targets,
        )
        page_part.metadata.update(extraction_info)
        semantic_dom_artifact = self.store.put_json_artifact(
            {"schema_version": self.workspace.NORMALIZATION_VERSION, "url": page_url,
             "title": title, "nodes": semantic_nodes(_document),
             "structured_source_artifact_id": extraction_info["structured_source_artifact_id"]},
            kind="html_semantic_dom", source_id=self.workspace.manifest.source_id,
            metadata={"url": page_url}, preserve_text_integrity=True,
        )
        page_part.metadata["semantic_dom_artifact_id"] = semantic_dom_artifact.artifact_id
        self.workspace.allowed_artifacts.add(semantic_dom_artifact.artifact_id)
        for part, text in sections:
            part.metadata["navigation_depth"] = navigation_depth
            part.locator["navigation_depth"] = navigation_depth
            if capability_branch:
                part.metadata["capability_branch"] = capability_branch
                part.locator["capability_branch"] = capability_branch
            if parent_url:
                part.metadata["navigation_parent_url"] = parent_url
            duplicate_of = next(
                (
                    existing_id
                    for existing_id, existing_text in self.workspace.part_text.items()
                    if text
                    and existing_text == text
                    and self.workspace.parts[existing_id].kind
                    in {"html_section", "html_code", "html_navigation_group"}
                    and self.workspace.parts[existing_id].metadata.get(
                        "evidence_default", True
                    )
                ),
                None,
            )
            if duplicate_of:
                part.metadata["evidence_default"] = False
                part.metadata["exact_duplicate_of"] = duplicate_of
            if text:
                self.workspace.add_part(part, text)
            else:
                self.workspace.add_part(part)
            self.workspace.inventoried_parts.add(part.part_id)
        page_part.metadata["section_root_part_ids"] = [
            part.part_id
            for part, _ in sections
            if part.parent_part_id == page_part_id
        ]
        markdown_lines = [f"# {title}"]
        for part, text in sections:
            markdown_lines.extend((f"## {part.label}", text))
        markdown_artifact = self.store.put_artifact(
            "\n\n".join(markdown_lines).encode("utf-8"),
            kind="html_markdown_projection",
            media_type="text/markdown",
            suffix=".md",
            source_id=self.workspace.manifest.source_id,
            metadata={"url": page_url},
        )
        page_part.metadata["markdown_artifact_id"] = markdown_artifact.artifact_id
        self.workspace.allowed_artifacts.add(markdown_artifact.artifact_id)
        canonical_page_artifact = self.store.put_json_artifact(
            {
                "schema_version": "canonical-page.v2",
                "source_id": self.workspace.manifest.source_id,
                "url": page_url,
                "title": title,
                "navigation_depth": navigation_depth,
                "capability_branch": capability_branch,
                "candidate_units": [
                    {
                        "part_id": part.part_id,
                        "label": part.label,
                        "kind": part.kind,
                        "ordinal": part.ordinal,
                        "parent_part_id": part.parent_part_id,
                        "heading_level": part.locator.get("heading_level"),
                        "section_path": part.locator.get("section_path"),
                        "text_artifact_id": part.text_artifact_id,
                        "content_sha256": part.content_sha256,
                        "char_count": part.char_count,
                        "evidence_candidate": part.metadata.get(
                            "evidence_default", True
                        ),
                        "exact_duplicate_of": part.metadata.get(
                            "exact_duplicate_of"
                        ),
                    }
                    for part, _ in sections
                ],
            },
            kind="html_canonical_page",
            source_id=self.workspace.manifest.source_id,
            metadata={"url": page_url, "candidate_count": len(sections)},
            preserve_text_integrity=True,
        )
        page_part.metadata["canonical_page_artifact_id"] = (
            canonical_page_artifact.artifact_id
        )
        self.workspace.allowed_artifacts.add(canonical_page_artifact.artifact_id)
        page_payload = {
            **extraction_info,
            "url": page_url,
            "title": title,
            "page_part_id": page_part_id,
            "artifact_id": artifact_id,
            "semantic_dom_artifact_id": semantic_dom_artifact.artifact_id,
            "markdown_artifact_id": markdown_artifact.artifact_id,
            "canonical_page_artifact_id": canonical_page_artifact.artifact_id,
            "section_count": len(sections),
            "char_count": sum(len(text) for _, text in sections),
            "navigation_depth": navigation_depth,
            "capability_branch": capability_branch,
            "authored_navigation_groups": list(
                dict.fromkeys(authored_navigation_groups or [])
            ),
            "navigation_parent_url": parent_url,
        }
        return soup, page_part, page_payload, sections

    def _html_inventory(self, max_pages: int | None = None) -> dict[str, Any]:
        if self.workspace.manifest.primary_modality != "html":
            raise ValueError("Current Source is not HTML")
        if self.workspace.manifest.source_metadata.get("extraction_scope", {}).get("mode") == "selected":
            from .html_scope import selected_html_inventory
            return selected_html_inventory(self, max_pages=max_pages)
        configured_child_budget = max(0, int(self.settings.max_html_pages))
        if self.workspace.manifest.source_origin == "websearch":
            configured_child_budget = max(
                0,
                min(configured_child_budget, int(self.settings.websearch_html_max_pages)),
            )
        requested_budget = (
            configured_child_budget if max_pages is None else max(0, int(max_pages))
        )
        child_page_budget = min(requested_budget, configured_child_budget)
        raw_id = self._require_raw_artifact()
        root_data = self.store.artifact_path(raw_id).read_bytes()
        root_url = self.workspace.manifest.final_url or self.workspace.manifest.origin_url
        synthetic_root = root_url or f"artifact://{self.workspace.manifest.source_id}/root"
        navigation_scope = root_url
        pages: list[dict[str, Any]] = []
        navigation_gaps: list[dict[str, Any]] = []
        version_sample: list[str] = [self.workspace.manifest.display_name]

        root_soup, root_part, root_page, root_sections = self._materialize_html_page(
            page_url=synthetic_root,
            data=root_data,
            artifact_id=raw_id,
            ordinal=1,
            navigation_depth=0,
            capability_branch=None,
            parent_url=None,
        )
        pages.append(root_page)
        version_sample.append(root_page["title"])
        version_sample.extend(text for _, text in root_sections[:4])
        if root_url:
            inferred_scope = self._infer_tutorial_scope(root_soup, root_url)
            if inferred_scope and inferred_scope != root_url:
                navigation_scope = inferred_scope
                self.store.emit(
                    "html_tutorial_scope_expanded",
                    {"root_url": root_url, "tutorial_scope": inferred_scope},
                    source_id=self.workspace.manifest.source_id,
                    agent=self.agent_name,
                )

        root_branches: list[dict[str, Any]] = []
        if navigation_scope and synthetic_root.startswith(("http://", "https://", "file://")):
            root_branches = self._navigation_branches(
                root_soup,
                synthetic_root,
                navigation_scope,
            )
        root_groups = self._navigation_display_groups(
            root_soup,
            synthetic_root,
            navigation_scope or synthetic_root,
            root_branches,
        )
        root_structure = self._root_structure_projection(
            root_soup,
            synthetic_root,
            navigation_scope or synthetic_root,
            root_groups,
        )
        root_structure_artifact = self.store.put_json_artifact(
            root_structure,
            kind="html_root_structure",
            source_id=self.workspace.manifest.source_id,
            metadata={"url": synthetic_root},
            preserve_text_integrity=True,
        )
        self.workspace.root_structure_artifact_id = root_structure_artifact.artifact_id
        self.workspace.allowed_artifacts.add(root_structure_artifact.artifact_id)
        root_part.metadata["root_structure_artifact_id"] = (
            root_structure_artifact.artifact_id
        )
        root_page["root_structure_artifact_id"] = root_structure_artifact.artifact_id
        self.store.emit(
            "html_root_structure_materialized",
            {
                "artifact_id": root_structure_artifact.artifact_id,
                "heading_count": len(root_structure["headings"]),
                "link_tree_count": len(root_structure["nested_link_trees"]),
            },
            source_id=self.workspace.manifest.source_id,
            agent=self.agent_name,
        )
        root_links = [
            link["url"]
            for branch in root_branches
            for link in branch["links"]
        ]
        root_part.metadata["controlled_navigation"] = root_links
        root_part.metadata["navigation_branches"] = [
            {
                "branch_id": branch["branch_id"],
                "label": branch["label"],
                "link_count": len(branch["links"]),
                "links": [link["url"] for link in branch["links"]],
            }
            for branch in root_branches
        ]
        root_part.metadata["navigation_groups"] = root_groups
        root_page["controlled_navigation"] = root_links
        root_page["navigation_branches"] = root_part.metadata["navigation_branches"]
        root_page["navigation_groups"] = root_groups
        navigation_sampling = {
            "strategy": "seeded_random_per_branch_depth.v1",
            "seed": _stable_id(
                "navseed", self.store.run_id, self.workspace.manifest.source_id,
                synthetic_root, length=32,
            ),
        }
        root_part.metadata["navigation_sampling"] = navigation_sampling
        root_page["navigation_sampling"] = navigation_sampling
        self.store.emit(
            "html_navigation_sampling_configured",
            navigation_sampling,
            source_id=self.workspace.manifest.source_id,
            agent=self.agent_name,
        )
        self._record_html_link_observations(
            root_soup, synthetic_root, navigation_scope or synthetic_root,
            root_part, root_page,
        )

        branch_order = [branch["branch_id"] for branch in root_branches]
        branch_stats: dict[str, dict[str, Any]] = {
            branch["branch_id"]: {
                "branch_id": branch["branch_id"],
                "label": branch["label"],
                "discovered_by_depth": Counter(),
                "fetched_by_depth": Counter(),
                "failed_by_depth": Counter(),
            }
            for branch in root_branches
        }
        frontier: dict[str, deque[dict[str, Any]]] = {
            branch_id: deque() for branch_id in branch_order
        }
        discovered_urls: set[str] = {synthetic_root}
        attempted_urls: set[str] = {synthetic_root}
        seen_urls: set[str] = {synthetic_root}

        def enqueue_links(
            target_frontier: dict[str, deque[dict[str, Any]]],
            branch_id: str,
            links: list[dict[str, str]],
            *,
            depth: int,
            parent_url: str,
        ) -> None:
            queue = target_frontier.setdefault(branch_id, deque())
            for link in links:
                url = link["url"]
                if url in discovered_urls:
                    continue
                discovered_urls.add(url)
                queue.append(
                    {
                        "url": url,
                        "text": link.get("text", ""),
                        "depth": depth,
                        "parent_url": parent_url,
                        "authored_navigation_groups": list(
                            link.get("authored_navigation_groups", [])
                        ),
                    }
                )
                branch_stats[branch_id]["discovered_by_depth"][depth] += 1

        for branch in root_branches:
            enqueue_links(
                frontier,
                branch["branch_id"],
                branch["links"],
                depth=1,
                parent_url=synthetic_root,
            )

        child_page_count = 0
        navigation_depth = 1
        pending_frontiers: list[dict[str, deque[dict[str, Any]]]] = []
        while any(frontier.values()) and child_page_count < child_page_budget:
            # Rank the complete branch/depth pool, including links discovered
            # from different parents. A run-scoped hash gives each URL a stable
            # pseudorandom priority without favoring its position in the manual.
            for branch_id, queue in frontier.items():
                frontier[branch_id] = deque(sorted(
                    queue,
                    key=lambda item: _stable_id(
                        "navpriority", navigation_sampling["seed"], branch_id,
                        str(navigation_depth), item["url"], length=64,
                    ),
                ))
            next_frontier: dict[str, deque[dict[str, Any]]] = {
                branch_id: deque() for branch_id in branch_order
            }
            while any(frontier.values()) and child_page_count < child_page_budget:
                progressed = False
                for branch_id in branch_order:
                    queue = frontier[branch_id]
                    while queue and (
                        queue[0]["url"] in attempted_urls or queue[0]["url"] in seen_urls
                    ):
                        queue.popleft()
                    if not queue:
                        continue
                    if self.store.cancellation_requested():
                        raise RuntimeError("run_cancelled")
                    candidate = queue.popleft()
                    candidate_url = candidate["url"]
                    attempted_urls.add(candidate_url)
                    progressed = True
                    fetched = self.workspace.acquisition.fetch_html_child(
                        self.workspace.manifest.source_id,
                        navigation_scope or synthetic_root,
                        candidate_url,
                    )
                    if fetched["status"] != "succeeded":
                        reason = str(fetched.get("reason", "fetch_failed"))
                        navigation_gaps.append(
                            {
                                "url": candidate_url,
                                "reason": reason,
                                "error_type": fetched.get("error_type"),
                                "message": str(fetched.get("message") or "")[:1_600],
                                "navigation_depth": navigation_depth,
                                "capability_branch": branch_stats[branch_id]["label"],
                            }
                        )
                        branch_stats[branch_id]["failed_by_depth"][navigation_depth] += 1
                        continue
                    final_url = str(fetched.get("final_url") or candidate_url)
                    if final_url in seen_urls:
                        continue
                    seen_urls.update({candidate_url, final_url})
                    child_id = str(fetched["artifact_id"])
                    self.workspace.allowed_artifacts.add(child_id)
                    soup, page_part, page_payload, sections = self._materialize_html_page(
                        page_url=final_url,
                        data=fetched["data"],
                        artifact_id=child_id,
                        ordinal=len(pages) + 1,
                        navigation_depth=navigation_depth,
                        capability_branch=branch_stats[branch_id]["label"],
                        parent_url=candidate["parent_url"],
                        authored_navigation_groups=candidate.get(
                            "authored_navigation_groups", []
                        ),
                    )
                    child_page_count += 1
                    branch_stats[branch_id]["fetched_by_depth"][navigation_depth] += 1
                    if len(pages) < 3:
                        version_sample.append(page_payload["title"])
                        version_sample.extend(text for _, text in sections[:4])
                    candidate_records = self._navigation_link_records(
                        soup,
                        final_url,
                        navigation_scope or synthetic_root,
                    )
                    candidate_links = [record["url"] for record in candidate_records]
                    page_part.metadata["controlled_navigation"] = candidate_links
                    page_payload["controlled_navigation"] = candidate_links
                    self._record_html_link_observations(
                        soup, final_url, navigation_scope or synthetic_root,
                        page_part, page_payload,
                    )
                    pages.append(page_payload)
                    enqueue_links(
                        next_frontier,
                        branch_id,
                        [
                            {
                                "url": record["url"],
                                "text": record["text"],
                                "authored_navigation_groups": [
                                    str(group.get("label") or "")
                                    for group in record.get("authored_navigation_groups", [])
                                    if group.get("label")
                                ],
                            }
                            for record in candidate_records
                        ],
                        depth=navigation_depth + 1,
                        parent_url=final_url,
                    )
                    if child_page_count >= child_page_budget:
                        break
                if not progressed:
                    break
            if child_page_count >= child_page_budget:
                pending_frontiers = [frontier, next_frontier]
                break
            frontier = next_frontier
            navigation_depth += 1

        if not pending_frontiers:
            pending_frontiers = [frontier]
        pending_urls = list(
            dict.fromkeys(
                item["url"]
                for pending in pending_frontiers
                for queue in pending.values()
                for item in queue
                if item["url"] not in attempted_urls and item["url"] not in seen_urls
            )
        )
        truncated = bool(pending_urls)
        if truncated:
            self.workspace.gaps.append(
                "The HTML child-page budget was exhausted: "
                f"the {child_page_budget}-page budget covered {len(root_branches)} top-level capability branches; "
                f"{len(discovered_urls) - 1} child pages were discovered, {child_page_count} were materialized, "
                f"and {len(pending_urls)} discovered pages remain. Pages were scheduled breadth-first "
                "and round-robin across capability branches, with seeded random order within each branch and depth."
            )
        if navigation_gaps:
            self.workspace.gaps.append(
                f"Acquisition or admission failed for {len(navigation_gaps)} HTML child pages; details are stored in the HTML inventory."
            )
            if child_page_count == 0:
                self.workspace.gaps.append(
                    "Only the HTML entry page was captured; linked child-page body content "
                    "is unavailable. The entry page and source links are retained as fallback."
                )

        branch_coverage: list[dict[str, Any]] = []
        for branch_id in branch_order:
            stats = branch_stats[branch_id]
            discovered_by_depth = {
                str(depth): count
                for depth, count in sorted(stats["discovered_by_depth"].items())
            }
            fetched_by_depth = {
                str(depth): count
                for depth, count in sorted(stats["fetched_by_depth"].items())
            }
            failed_by_depth = {
                str(depth): count
                for depth, count in sorted(stats["failed_by_depth"].items())
            }
            discovered = sum(discovered_by_depth.values())
            fetched = sum(fetched_by_depth.values())
            failed = sum(failed_by_depth.values())
            branch_coverage.append(
                {
                    "branch_id": branch_id,
                    "label": stats["label"],
                    "discovered_pages": discovered,
                    "fetched_pages": fetched,
                    "failed_pages": failed,
                    "unfetched_pages": max(0, discovered - fetched - failed),
                    "discovered_by_depth": discovered_by_depth,
                    "fetched_by_depth": fetched_by_depth,
                    "failed_by_depth": failed_by_depth,
                }
            )
        depth_values = sorted(
            {
                depth
                for stats in branch_stats.values()
                for key in ("discovered_by_depth", "fetched_by_depth", "failed_by_depth")
                for depth in stats[key]
            }
        )
        depth_coverage = [
            {
                "navigation_depth": depth,
                "discovered_pages": sum(
                    stats["discovered_by_depth"].get(depth, 0)
                    for stats in branch_stats.values()
                ),
                "fetched_pages": sum(
                    stats["fetched_by_depth"].get(depth, 0)
                    for stats in branch_stats.values()
                ),
                "failed_pages": sum(
                    stats["failed_by_depth"].get(depth, 0)
                    for stats in branch_stats.values()
                ),
            }
            for depth in depth_values
        ]

        self.workspace.inventory_completed = True
        self.workspace.update_document_version("\n".join(version_sample))
        payload = {
            "root_structure_artifact_id": self.workspace.root_structure_artifact_id,
            "page_count": len(pages),
            "root_page_count": 1,
            "child_page_count": child_page_count,
            "part_count": len(self.workspace.parts),
            "pages": pages,
            "navigation_gaps": navigation_gaps,
            "acquisition_complete": not truncated and not navigation_gaps,
            "truncated": truncated,
            "unfetched_discovered_urls": pending_urls,
            "page_budget_scope": "child_pages_only",
            "max_pages": child_page_budget,
            "max_child_pages": child_page_budget,
            "navigation_sampling": navigation_sampling,
            "capability_branch_count": len(root_branches),
            "branch_coverage": branch_coverage,
            "depth_coverage": depth_coverage,
        }
        artifact = self.store.put_json_artifact(
            payload,
            kind="html_inventory",
            source_id=self.workspace.manifest.source_id,
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(artifact.artifact_id)
        self.workspace.persist()
        return {"inventory_artifact_id": artifact.artifact_id, **payload}

    def _html_list_navigation(self, offset: int = 0, limit: int = 40) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            raise ValueError("Run html_inventory first")
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 50))
        all_pages = [
            {
                "part_id": part.part_id,
                "label": part.label,
                "url": part.locator.get("url"),
                "navigation_depth": part.metadata.get("navigation_depth", 0),
                "capability_branch": part.metadata.get("capability_branch"),
                "authored_navigation_groups": part.metadata.get(
                    "authored_navigation_groups", []
                ),
                "navigation_parent_url": part.metadata.get("navigation_parent_url"),
                "controlled_navigation": part.metadata.get("controlled_navigation", []),
                "link_role_counts": part.metadata.get("link_role_counts", {}),
                "link_observations_artifact_id": part.metadata.get("link_observations_artifact_id"),
                "navigation_branches": part.metadata.get("navigation_branches", []),
                "semantic_dom_artifact_id": part.metadata.get("semantic_dom_artifact_id"),
                "markdown_artifact_id": part.metadata.get("markdown_artifact_id"),
            }
            for part in self.workspace.parts.values()
            if part.kind == "html_page"
        ]
        pages = all_pages[offset : offset + limit]
        return {
            "page_count": len(all_pages),
            "offset": offset,
            "returned": len(pages),
            "truncated": offset + len(pages) < len(all_pages),
            "pages": pages,
        }

    def _semantic_dom_projection(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        root = soup.find("main") or soup.find("article") or soup.body or soup
        nodes: list[dict[str, Any]] = []
        heading_stack: list[dict[str, Any]] = []
        for ordinal, node in enumerate(
            root.find_all(
                [
                    "h1", "h2", "h3", "h4", "h5", "h6",
                    "summary", "p", "pre", "table", "li", "img", "a",
                ]
            ),
            start=1,
        ):
            if not isinstance(node, Tag) or node.find_parent(["script", "style", "noscript", "template"]):
                continue
            payload: dict[str, Any] = {"ordinal": ordinal, "tag": node.name}
            if node.name and node.name.startswith("h") and len(node.name) == 2:
                level = int(node.name[1])
                while heading_stack and int(heading_stack[-1]["level"]) >= level:
                    heading_stack.pop()
                payload["heading_level"] = level
                payload["parent_heading_ordinal"] = (
                    int(heading_stack[-1]["ordinal"]) if heading_stack else None
                )
            if node.name == "img":
                payload["src"] = str(node.get("src") or "")
                payload["alt"] = str(node.get("alt") or "")
            elif node.name == "a":
                payload["href"] = str(node.get("href") or "")
                payload["text"] = _compact_text(node.get_text(" ", strip=True))
            elif node.name == "summary":
                payload["text"] = _compact_text(node.get_text(" ", strip=True))
                payload["semantic_container"] = "details"
                payload["container_id"] = (
                    str(node.parent.get("id") or "")
                    if isinstance(node.parent, Tag) and node.parent.name == "details"
                    else ""
                )
            elif node.name == "pre":
                payload["text"] = node.get_text("\n", strip=False).replace("\x00", "").strip("\n")
            elif node.name == "table":
                payload["rows"] = [
                    [_compact_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                    for row in node.find_all("tr")
                ]
                payload["text"] = _compact_text(node.get_text("\n", strip=True))
            else:
                payload["text"] = _compact_text(node.get_text("\n" if node.name == "table" else " ", strip=True))
            if node.name in HTML_HEADING_TAGS:
                heading_stack.append(
                    {
                        "level": int(node.name[1]),
                        "ordinal": ordinal,
                        "title": payload.get("text", ""),
                    }
                )
            payload["heading_path"] = [
                str(item["title"]) for item in heading_stack if item.get("title")
            ]
            nodes.append(payload)
        return nodes

    @staticmethod
    def _structural_label(node: Tag | None) -> str:
        """Return an authored label without heading permalink chrome."""

        if not isinstance(node, Tag):
            return ""
        fragments: list[str] = []
        for value in node.descendants:
            if not isinstance(value, NavigableString) or not str(value).strip():
                continue
            owner = value.parent
            if isinstance(owner, Tag):
                permalink = owner.find_parent("a") if owner.name != "a" else owner
                if isinstance(permalink, Tag) and is_heading_permalink(permalink):
                    continue
            fragments.append(str(value))
        label = _compact_text(" ".join(fragments))
        return re.sub(r"\s*[¶#]+\s*$", "", label).strip()

    def _root_structure_projection(
        self,
        soup: BeautifulSoup,
        page_url: str,
        root_url: str,
        visual_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a compact, site-neutral view of the authored root structure.

        This projection preserves headings, disclosure containers, visual groups,
        nested link lists, fragments, DOM order, roles, and classes. It deliberately
        does not decide which level should become Evidence; that is the Architect's
        semantic responsibility.
        """

        root = (
            soup.find("main")
            or soup.find("article", attrs={"role": "main"})
            or soup.find("article")
            or soup.body
            or soup
        )
        base_url = html_base_url(soup, page_url)
        heading_stack: list[dict[str, Any]] = []
        headings: list[dict[str, Any]] = []
        heading_by_tag_id: dict[int, dict[str, Any]] = {}
        for ordinal, heading in enumerate(root.find_all(sorted(HTML_HEADING_TAGS)), start=1):
            if not isinstance(heading, Tag):
                continue
            level = int(str(heading.name)[1])
            while heading_stack and int(heading_stack[-1]["level"]) >= level:
                heading_stack.pop()
            label = self._structural_label(heading)
            if not label:
                continue
            node = {
                "node_id": _stable_id(
                    "rsv", page_url, "heading", str(ordinal), label, length=16
                ),
                "kind": "heading",
                "tag": heading.name,
                "level": level,
                "label": label,
                "dom_id": str(heading.get("id") or "") or None,
                "parent_node_id": heading_stack[-1]["node_id"] if heading_stack else None,
                "order": ordinal,
            }
            headings.append(node)
            heading_by_tag_id[id(heading)] = node
            heading_stack.append(node)

        def absolute_link(anchor: Tag | None) -> dict[str, Any]:
            if not isinstance(anchor, Tag):
                return {}
            href = str(anchor.get("href") or "").strip()
            if not href or href.startswith(("javascript:", "mailto:", "data:")):
                return {}
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https", "file"}:
                return {}
            document_url = urlunparse(parsed._replace(fragment=""))
            projected = {
                "label": self._structural_label(anchor),
                "href": absolute,
                "fragment": parsed.fragment or None,
                "link_role": classify_html_link(anchor)[0],
            }
            if not self.workspace.acquisition.is_within_tutorial_scope(
                root_url, document_url
            ):
                projected["in_scope"] = False
            return projected

        list_tags = ["ul", "ol", "dl"]

        def item_projection(item: Tag, owner: Tag) -> dict[str, Any]:
            # Inline wrappers do not change ownership. Never borrow a nested
            # item's anchor as the label/link of its parent.
            direct_anchor = next(
                (anchor for anchor in item.find_all("a", href=True)
                 if anchor.find_parent(list_tags) is owner),
                None,
            )
            link = absolute_link(direct_anchor)
            direct_text = _compact_text(
                " ".join(
                    str(value)
                    for value in item.find_all(string=True)
                    if str(value).strip() and value.find_parent(list_tags) is owner
                )
            )
            label = str(link.get("label") or direct_text or "List item")
            children: list[dict[str, Any]] = []
            for child_list in item.find_all(list_tags):
                if child_list.find_parent(list_tags) is owner:
                    children.extend(list_items(child_list))
            return {
                "label": label,
                **link,
                "children": children,
            }

        def list_items(list_node: Tag) -> list[dict[str, Any]]:
            if list_node.name != "dl":
                return [item_projection(item, list_node)
                        for item in list_node.find_all("li", recursive=False)]
            # Definition-list contents associate each dt with the following dd
            # subtree. Plain prose definitions retain their non-directory roles.
            items: list[dict[str, Any]] = []
            for child in list_node.find_all(["dt", "dd"], recursive=False):
                if child.name == "dt":
                    items.append(item_projection(child, list_node))
                elif items:
                    for child_list in child.find_all(list_tags):
                        if child_list.find_parent(list_tags) is list_node:
                            items[-1]["children"].extend(list_items(child_list))
            return items

        link_trees: list[dict[str, Any]] = []
        top_lists = [
            node
            for node in root.find_all(list_tags)
            if isinstance(node, Tag) and node.find_parent(list_tags) is None
        ]
        for list_index, list_node in enumerate(top_lists, start=1):
            items = list_items(list_node)
            if not items:
                continue
            if self._is_html_navigation_chrome(list_node) and (
                not is_authored_contents(list_node)
                or list_node.find_parent(["header", "footer", "aside"])
            ):
                continue
            links = [
                anchor
                for anchor in list_node.find_all("a", href=True)
                if absolute_link(anchor)
            ]
            if not links:
                continue
            details = list_node.find_parent("details")
            summary = details.find("summary", recursive=False) if isinstance(details, Tag) else None
            previous_heading = list_node.find_previous(sorted(HTML_HEADING_TAGS))
            heading_context = (
                heading_by_tag_id.get(id(previous_heading))
                if isinstance(previous_heading, Tag)
                else None
            )
            parent = list_node.parent if isinstance(list_node.parent, Tag) else None
            caption = list_node.find_previous_sibling(
                lambda tag: isinstance(tag, Tag)
                and tag.name in {"p", "div", "span"}
            )
            context_label = (
                self._structural_label(summary)
                or self._structural_label(caption)
                or (str(heading_context.get("label")) if heading_context else "")
            )
            link_trees.append(
                {
                    "tag": list_node.name,
                    "context_label": context_label or None,
                    "heading_node_id": (
                        heading_context.get("node_id") if heading_context else None
                    ),
                    "container_tag": parent.name if parent is not None else None,
                    "container_classes": (
                        [str(item) for item in parent.get("class", [])]
                        if parent is not None
                        else []
                    ),
                    "role": (
                        str(parent.get("role") or "") or None
                        if parent is not None
                        else None
                    ),
                    "link_count": len(links),
                    "authored_contents": is_authored_contents(list_node),
                    "items": items,
                }
            )

        disclosures: list[dict[str, Any]] = []
        for index, details in enumerate(root.find_all("details"), start=1):
            if not isinstance(details, Tag):
                continue
            summary = details.find("summary", recursive=False)
            links = [absolute_link(anchor) for anchor in details.find_all("a", href=True)]
            links = [link for link in links if link]
            disclosure = {
                    "node_id": _stable_id(
                        "rsv", page_url, "details", str(index), length=16
                    ),
                    "kind": "disclosure",
                    "label": self._structural_label(summary),
                    "dom_id": str(details.get("id") or "") or None,
                    "link_count": len(links),
                }
            # Nested lists already carry every link with its authored depth. Keep
            # direct disclosure links only when no list represents them, avoiding
            # a second full copy for ChimeraX-style expandable catalogs.
            if details.find(list_tags) is None:
                disclosure["links"] = links
            disclosures.append(disclosure)

        visual_group_summaries = [
            {
                "group_id": group.get("group_id"),
                "label": group.get("label"),
                "labels": group.get("labels", []),
                "link_count": group.get("link_count", 0),
                "container_tag": group.get("container_tag"),
                "container_classes": group.get("container_classes", []),
            }
            for group in visual_groups
        ]

        return {
            "schema_version": "root-structure.v2",
            "source_id": self.workspace.manifest.source_id,
            "document_title": self._structural_label(soup.title) or self.workspace.manifest.display_name,
            "root_url": page_url,
            "primary_region_tag": root.name if isinstance(root, Tag) else None,
            "headings": headings,
            "disclosures": disclosures,
            "nested_link_trees": link_trees,
            "python_visual_group_candidates": visual_group_summaries,
            "instructions": (
                "This is a lossless structural view, not an Evidence decision. Infer the "
                "meaningful topic hierarchy from authored headings, containers, and list depth. "
                "Only directory_entry links support page parent-child relationships. "
                "Cross references, heading permalinks, and previous/next controls do not "
                "make their destinations children of the surrounding section."
            ),
        }

    def _is_html_navigation_chrome(self, node: Tag) -> bool:
        chrome_tokens = {
            "breadcrumb",
            "breadcrumbs",
            "footer",
            "header",
            "navbar",
            "site-footer",
            "site-header",
            "site-nav",
            "toc",
        }
        for parent in [node, *[item for item in node.parents if isinstance(item, Tag)]]:
            if parent.name in {
                "script",
                "style",
                "noscript",
                "template",
                "header",
                "footer",
                "nav",
            }:
                return True
            tokens = {
                str(parent.get("id") or "").strip().casefold(),
                *(str(value).strip().casefold() for value in parent.get("class", [])),
            }
            if tokens & chrome_tokens:
                return True
            role = str(parent.get("role") or "").strip().casefold()
            if role in {"navigation", "banner", "contentinfo", "doc-toc"}:
                return True
            if any(
                re.search(r"(?:^|[-_])(nav(?:igation)?|breadcrumb|footer|header)(?:$|[-_])", token)
                for token in tokens
                if token
            ):
                return True
        return False

    def _html_link_observations(
        self,
        soup: BeautifulSoup,
        page_url: str,
        root_url: str,
    ) -> list[dict[str, Any]]:
        content_root = soup.find("main") or soup.find("article") or soup.body or soup
        base_url = html_base_url(soup, page_url)
        document_url = urlunparse(urlparse(page_url)._replace(fragment=""))
        heading_path: dict[int, str] = {}
        records: list[dict[str, Any]] = []
        for node in content_root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "a"],
        ):
            if not isinstance(node, Tag):
                continue
            if node.name and re.fullmatch(r"h[1-6]", node.name):
                if self._is_html_navigation_chrome(node):
                    continue
                level = int(node.name[1])
                label = self._structural_label(node)
                if not label:
                    continue
                heading_path[level] = label
                for deeper in [value for value in heading_path if value > level]:
                    heading_path.pop(deeper, None)
                continue
            href = str(node.get("href", "")).strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https", "file"}:
                continue
            normalized = urlunparse(parsed._replace(fragment=""))
            role, reason = classify_html_link(node)
            if normalized == document_url and role != "heading_permalink":
                role, reason = "local_anchor", "same_document"
            in_scope = self.workspace.acquisition.is_within_tutorial_scope(root_url, normalized)
            parsed_path = Path(normalized.split("?", 1)[0])
            chrome = self._is_html_navigation_chrome(node)
            local_contents = is_authored_contents(node) and not node.find_parent(
                ["header", "footer", "aside"]
            )
            crawl_candidate = (
                role == "directory_entry" and in_scope and normalized != document_url
                and parsed_path.suffix.lower() in {"", ".html", ".htm"}
                and (not chrome or local_contents)
            )
            if chrome and not local_contents and role == "directory_entry":
                role, reason = "site_navigation", "navigation_chrome"
            disclosure_groups: list[dict[str, str | None]] = []
            for details in reversed(node.find_parents("details")):
                summary = details.find("summary", recursive=False)
                label = (
                    self._structural_label(summary)
                    if isinstance(summary, Tag)
                    else ""
                )
                if not label:
                    continue
                disclosure_groups.append(
                    {
                        "label": label,
                        "dom_id": str(details.get("id") or "") or None,
                    }
                )
            records.append(
                {
                    "url": normalized,
                    "href": absolute,
                    "raw_href": href,
                    "fragment": parsed.fragment or None,
                    "text": _compact_text(node.get_text(" ", strip=True)),
                    "link_role": role,
                    "role_reason": reason,
                    "in_scope": in_scope,
                    "crawl_candidate": crawl_candidate,
                    "heading_path": dict(heading_path),
                    "authored_navigation_groups": disclosure_groups,
                    "record_order": len(records),
                }
            )

        return records

    def _record_html_link_observations(
        self, soup: BeautifulSoup, page_url: str, root_url: str,
        page_part: SourcePart, page_payload: dict[str, Any],
    ) -> None:
        observations = self._html_link_observations(soup, page_url, root_url)
        counts = dict(Counter(record["link_role"] for record in observations))
        artifact = self.store.put_json_artifact(
            {"url": page_url, "base_url": html_base_url(soup, page_url),
             "link_role_counts": counts, "links": [
                 {**record, "heading_path": {
                     str(level): label for level, label in record["heading_path"].items()
                 }} for record in observations
             ]},
            kind="html_link_observations", source_id=self.workspace.manifest.source_id,
            metadata={"url": page_url}, preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(artifact.artifact_id)
        for metadata in (page_part.metadata, page_payload):
            metadata["link_role_counts"] = counts
            metadata["link_observations_artifact_id"] = artifact.artifact_id
        self.store.emit(
            "html_navigation_classified",
            {"url": page_url, "artifact_id": artifact.artifact_id,
             "link_role_counts": counts,
             "directory_candidate_count": sum(record["crawl_candidate"] for record in observations)},
            source_id=self.workspace.manifest.source_id, agent=self.agent_name,
        )

    def _navigation_link_records(
        self, soup: BeautifulSoup, page_url: str, root_url: str,
    ) -> list[dict[str, Any]]:
        records = [record for record in self._html_link_observations(soup, page_url, root_url)
                   if record["crawl_candidate"]]

        # A link can occur in introductory prose and later inside an authored
        # disclosure group. Prefer the grouped occurrence so repeated links do
        # not create a synthetic Overview branch or starve a visible group.
        selected: dict[str, dict[str, Any]] = {}
        memberships: dict[str, list[dict[str, str | None]]] = {}
        for record in records:
            url = str(record["url"])
            groups = list(record.get("authored_navigation_groups", []))
            known = memberships.setdefault(url, [])
            for group in groups:
                if group not in known:
                    known.append(group)
            if url not in selected:
                selected[url] = record
                continue
            if (
                not selected[url].get("authored_navigation_groups")
                and groups
            ):
                selected[url] = record
        chosen = sorted(
            selected.values(), key=lambda item: int(item.get("record_order", 0))
        )
        for record in chosen:
            primary_groups = list(record.get("authored_navigation_groups", []))
            record["primary_authored_navigation_group"] = (
                str(primary_groups[-1].get("label") or "")
                if primary_groups
                else ""
            )
            record["authored_navigation_groups"] = memberships[str(record["url"])]
            record.pop("record_order", None)
        return chosen

    def _navigation_branches(
        self,
        soup: BeautifulSoup,
        page_url: str,
        root_url: str,
    ) -> list[dict[str, Any]]:
        records = self._navigation_link_records(soup, page_url, root_url)
        labels_with_links = {
            _compact_text(str(group.get("label") or ""))
            for record in records
            for group in list(record.get("authored_navigation_groups", []))
            if _compact_text(str(group.get("label") or ""))
        }
        content_root = soup.find("main") or soup.find("article") or soup.body or soup
        disclosure_labels: list[str] = []
        for details in content_root.find_all("details"):
            if not isinstance(details, Tag) or self._is_html_navigation_chrome(details):
                continue
            summary = details.find("summary", recursive=False)
            label = (
                self._structural_label(summary)
                if isinstance(summary, Tag)
                else ""
            )
            if label in labels_with_links and label not in disclosure_labels:
                disclosure_labels.append(label)
        for record in records:
            groups = list(record.get("authored_navigation_groups", []))
            if not groups:
                continue
            label = _compact_text(
                str(record.get("primary_authored_navigation_group") or "")
            )
            if label and label not in disclosure_labels:
                disclosure_labels.append(label)
        if len(disclosure_labels) >= 2:
            has_ungrouped = any(
                not record.get("authored_navigation_groups") for record in records
            )
            labels = (["Overview"] if has_ungrouped else []) + disclosure_labels
            branches = [
                {
                    "branch_id": f"branch_{index:03d}",
                    "label": label,
                    "links": [],
                }
                for index, label in enumerate(labels, start=1)
            ]
            branches_by_label = {branch["label"]: branch for branch in branches}
            for record in records:
                groups = list(record.get("authored_navigation_groups", []))
                label = (
                    _compact_text(
                        str(record.get("primary_authored_navigation_group") or "")
                    )
                    if groups
                    else "Overview"
                )
                branch = branches_by_label.get(label)
                if branch is None:
                    continue
                branch["links"].append(
                    {
                        "url": record["url"],
                        "text": record["text"],
                        "authored_navigation_groups": [
                            str(group.get("label") or "")
                            for group in groups
                            if group.get("label")
                        ],
                    }
                )
            return [branch for branch in branches if branch["links"]]

        heading_levels = sorted(
            {
                level
                for record in records
                for level, label in record["heading_path"].items()
                if label
            }
        )
        branch_level: int | None = None
        for level in heading_levels:
            labels = {
                record["heading_path"].get(level)
                for record in records
                if record["heading_path"].get(level)
            }
            if len(labels) >= 2:
                branch_level = level
                break
        if branch_level is None and heading_levels:
            branch_level = max(heading_levels)

        labels: list[str] = []
        if branch_level is not None:
            for heading in content_root.find_all(f"h{branch_level}"):
                if not isinstance(heading, Tag) or self._is_html_navigation_chrome(heading):
                    continue
                label = self._structural_label(heading)
                if label and label not in labels:
                    labels.append(label)
        fallback_label = "Overview"
        if any(
            not record["heading_path"].get(branch_level) if branch_level is not None else True
            for record in records
        ):
            labels.insert(0, fallback_label)
        if not labels:
            labels = [fallback_label]

        branches: list[dict[str, Any]] = [
            {
                "branch_id": f"branch_{index:03d}",
                "label": label,
                "links": [],
            }
            for index, label in enumerate(labels, start=1)
        ]
        branches_by_label = {branch["label"]: branch for branch in branches}
        assigned_urls: set[str] = set()
        for record in records:
            label = (
                record["heading_path"].get(branch_level, fallback_label)
                if branch_level is not None
                else fallback_label
            )
            branch = branches_by_label.get(label) or branches_by_label[fallback_label]
            if record["url"] in assigned_urls:
                continue
            assigned_urls.add(record["url"])
            branch["links"].append({"url": record["url"], "text": record["text"]})
        return branches

    def _navigation_display_groups(
        self,
        soup: BeautifulSoup,
        page_url: str,
        root_url: str,
        branches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Group authored navigation headings by their nearest visual container.

        A card, table cell, section, or nav panel may contain several headings.
        The heading-level branches remain available for crawl allocation; this
        projection preserves the coarser grouping a reader sees on the root page.
        """

        content_root = soup.find("main") or soup.find("article") or soup.body or soup
        base_url = html_base_url(soup, page_url)
        directory_urls = {link["url"] for branch in branches for link in branch.get("links", [])}
        heading_tags = [
            tag
            for tag in content_root.find_all(
                [*sorted(HTML_SECTION_HEADING_TAGS), "summary"]
            )
            if isinstance(tag, Tag)
            and not self._is_html_navigation_chrome(tag)
            and (
                tag.name != "summary"
                or (isinstance(tag.parent, Tag) and tag.parent.name == "details")
            )
        ]
        heading_by_label: dict[str, Tag] = {}
        for heading in heading_tags:
            label = self._structural_label(heading)
            if label:
                heading_by_label.setdefault(label.casefold(), heading)

        grouped: dict[str, dict[str, Any]] = {}
        group_order: list[str] = []
        for branch in branches:
            label = _compact_text(str(branch.get("label") or "Overview"))
            heading = heading_by_label.get(label.casefold())
            container: Tag | None = None
            if heading is not None:
                current = heading.parent
                while isinstance(current, Tag) and current is not content_root:
                    valid_link_found = False
                    for anchor in current.find_all("a", href=True):
                        normalized = self.workspace.acquisition.normalize_url(
                            base_url, str(anchor.get("href") or "")
                        )
                        if normalized in directory_urls:
                            valid_link_found = True
                            break
                    if valid_link_found and current.name in {
                        "article",
                        "aside",
                        "div",
                        "details",
                        "li",
                        "nav",
                        "section",
                        "td",
                    }:
                        container = current
                        break
                    current = current.parent
            key = f"container:{id(container)}" if container is not None else str(
                branch.get("branch_id")
            )
            if key not in grouped:
                group_order.append(key)
                grouped[key] = {
                    "group_id": f"group_{len(group_order):03d}",
                    "labels": [],
                    "branch_ids": [],
                    "links": [],
                    "container_tag": container.name if container is not None else None,
                    "container_classes": (
                        [str(item) for item in container.get("class", [])]
                        if container is not None
                        else []
                    ),
                }
            group = grouped[key]
            group["labels"].append(label)
            group["branch_ids"].append(str(branch.get("branch_id")))
            for link in branch.get("links", []):
                url = str(link.get("url") or "")
                if url and url not in group["links"]:
                    group["links"].append(url)

        return [
            {
                **grouped[key],
                "label": " / ".join(grouped[key]["labels"]),
                "link_count": len(grouped[key]["links"]),
            }
            for key in group_order
        ]

    def _navigation_links(self, soup: BeautifulSoup, page_url: str, root_url: str) -> list[str]:
        return [
            record["url"]
            for record in self._navigation_link_records(soup, page_url, root_url)
        ]

    def _infer_tutorial_scope(self, soup: BeautifulSoup, root_url: str) -> str:
        root = urlparse(root_url)
        if root.scheme != "file":
            return root_url
        groups: dict[tuple[str, str, str], list[str]] = {}
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href") or "").strip()
            parsed = urlparse(href)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            segments = [segment for segment in parsed.path.split("/") if segment]
            if not segments:
                continue
            key = (parsed.scheme, parsed.netloc.casefold(), segments[0])
            groups.setdefault(key, []).append(parsed.path or "/")
        if not groups:
            return root_url
        key, paths = max(groups.items(), key=lambda item: len(item[1]))
        if len(paths) < 3:
            return root_url
        common_path = posixpath.commonpath(paths)
        if common_path == "/" or not common_path:
            return root_url
        scope_path = common_path.rstrip("/") + "/"
        return urlunparse((key[0], key[1], scope_path, "", "", ""))

    def _decode_html(self, data: bytes) -> str:
        from bs4 import UnicodeDammit

        return UnicodeDammit(data, is_html=True).unicode_markup or data.decode("utf-8", errors="replace")
