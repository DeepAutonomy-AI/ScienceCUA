"""Persist parser-owned Source parts, coverage, and grounded Evidence."""
from __future__ import annotations

import hashlib
from typing import Any

from ..acquisition import AcquisitionService, detect_document_version, resolve_version
from ..config import Settings
from ..models import (
    MAX_EVIDENCE_IMAGE_REFS,
    Coverage,
    EvidenceExcerpt,
    EvidenceImageRef,
    EvidencePlan,
    EvidenceTree,
    SourceManifest,
    SourcePart,
    SourceResult,
    TargetProfile,
)
from ..storage import RunStore
from .text import _excerpt_ranges, _stable_id


class SourceWorkspace:
    NORMALIZATION_VERSION = "v13"

    def __init__(
        self,
        store: RunStore,
        settings: Settings,
        acquisition: AcquisitionService,
        target_profile: TargetProfile,
        manifest: SourceManifest,
    ):
        self.store = store
        self.settings = settings
        self.acquisition = acquisition
        self.target_profile = target_profile
        self.manifest = manifest
        if manifest.primary_modality == "html":
            self.NORMALIZATION_VERSION = "v16-html-structured.v11-random-navigation.v2"
        elif manifest.primary_modality == "pdf":
            from ..pdf_sourcepart.pipeline import NORMALIZATION_VERSION
            self.NORMALIZATION_VERSION = NORMALIZATION_VERSION
        self.parts: dict[str, SourcePart] = {}
        self.part_text: dict[str, str] = {}
        self.evidence: dict[str, EvidenceExcerpt] = {}
        self.allowed_artifacts: set[str] = set()
        if manifest.raw_artifact_id:
            self.allowed_artifacts.add(manifest.raw_artifact_id)
        self.inventory_completed = False
        self.inventoried_parts: set[str] = set()
        self.normalized_parts: set[str] = set()
        self.inspected_parts: set[str] = set()
        self.evidence_parts: set[str] = set()
        self.skipped: dict[str, str] = {}
        self.gaps: list[str] = []
        self.warnings: list[str] = list(manifest.warnings)
        self.structure_submission: dict[str, Any] | None = None
        self.curation_submission: dict[str, Any] | None = None
        self.curated_part_ids: list[str] = []
        self.deterministic_evidence_plan: dict[str, Any] | None = None
        self.evidence_skeleton_plan: EvidencePlan | None = None
        self.evidence_skeleton_plan_artifact_id: str | None = None
        self.evidence_skeleton_tree_artifact_id: str | None = None
        self.root_structure_artifact_id: str | None = None
        self.pdf_document_structure_artifact_id: str | None = None
        self.pdf_document_structure: dict[str, Any] | None = None
        self.evidence_tree: EvidenceTree | None = None
        self.evidence_tree_artifact_id: str | None = None
        self.final_result: SourceResult | None = None
        self.corpus_artifact_id: str | None = None
        self._persist()

    def add_part(self, part: SourcePart, text: str | None = None) -> None:
        self.parts[part.part_id] = part
        if text is not None:
            # This text is the parser's normalized representation, not an Agent rewrite.
            exact_text = text if self.manifest.primary_modality == "html" else text.replace("\x00", "")
            artifact = self.store.put_artifact(
                exact_text.encode("utf-8"),
                kind="normalized_source_text",
                media_type="text/plain",
                suffix=".txt",
                source_id=self.manifest.source_id,
                metadata={
                    "source_part_id": part.part_id,
                    "normalization_version": self.NORMALIZATION_VERSION,
                },
            )
            part.text_artifact_id = artifact.artifact_id
            part.content_sha256 = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
            part.char_count = len(exact_text)
            self.part_text[part.part_id] = exact_text
            self.normalized_parts.add(part.part_id)
        if part.artifact_id:
            self.allowed_artifacts.add(part.artifact_id)
        if part.text_artifact_id:
            self.allowed_artifacts.add(part.text_artifact_id)

    def add_evidence_excerpt(
        self,
        part_id: str,
        start_offset: int,
        end_offset: int,
        image_refs: list[EvidenceImageRef] | None = None,
    ) -> EvidenceExcerpt:
        if part_id not in self.parts:
            raise ValueError(f"Unknown source_part_id: {part_id}")
        part = self.parts[part_id]
        refs = image_refs or []
        evidence_refs = refs
        image_refs_overflow: dict[str, Any] | None = None
        if len(refs) > MAX_EVIDENCE_IMAGE_REFS:
            evidence_refs = refs[:MAX_EVIDENCE_IMAGE_REFS]
            overflow_refs = refs[MAX_EVIDENCE_IMAGE_REFS:]
            overflow_artifact = self.store.put_json_artifact(
                {
                    "source_id": self.manifest.source_id,
                    "source_part_id": part_id,
                    "image_ref_count_total": len(refs),
                    "image_refs_returned": MAX_EVIDENCE_IMAGE_REFS,
                    "image_refs_overflow_count": len(overflow_refs),
                    "image_refs_overflow": [
                        ref.model_dump(mode="json") for ref in overflow_refs
                    ],
                },
                kind="evidence_image_refs_overflow",
                source_id=self.manifest.source_id,
                metadata={
                    "source_part_id": part_id,
                    "image_ref_count_total": len(refs),
                    "image_refs_returned": MAX_EVIDENCE_IMAGE_REFS,
                    "image_refs_overflow_count": len(overflow_refs),
                },
            )
            self.allowed_artifacts.add(overflow_artifact.artifact_id)
            image_refs_overflow = {
                "image_ref_count_total": len(refs),
                "image_refs_returned": MAX_EVIDENCE_IMAGE_REFS,
                "image_refs_overflow_count": len(overflow_refs),
                "image_refs_overflow_artifact_id": overflow_artifact.artifact_id,
            }
            warning = (
                f"Evidence {part_id} has {len(refs)} image references; "
                f"the first {MAX_EVIDENCE_IMAGE_REFS} are in image_refs and "
                f"{len(overflow_refs)} overflow references are stored in "
                f"{overflow_artifact.artifact_id}."
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
        has_text = bool(part.text_artifact_id and part_id in self.part_text)
        if not has_text and not refs:
            raise ValueError("SourcePart has neither normalized text nor image references")
        if has_text:
            start = max(0, int(start_offset))
            end = min(len(self.part_text[part_id]), int(end_offset))
            if end <= start:
                raise ValueError("Evidence span must have a positive length")
            content = self.part_text[part_id][start:end]
            if not content.strip():
                raise ValueError("Evidence span cannot contain only whitespace")
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            start = 0
            end = 0
            content = ""
            content_sha256 = hashlib.sha256(
                "\x1f".join(ref.artifact_id for ref in refs).encode("ascii")
            ).hexdigest()
        identity = (
            self.manifest.source_id, part_id, str(start), str(end),
            content_sha256, *(ref.artifact_id for ref in refs),
        )
        evidence_id = self.store.canonical_id(
            "E", _stable_id("evi", *identity), *identity
        )
        locator = dict(part.locator)
        if has_text:
            locator.update({"char_start": start, "char_end": end})
        if image_refs_overflow:
            locator.update(image_refs_overflow)
        if part.metadata.get("deterministic_projection"):
            extraction_method = "deterministic_structural_group"
        elif part.metadata.get("curated_projection"):
            extraction_method = "model_curated_source_unit"
        else:
            extraction_method = (
                "multimodal_source_unit" if evidence_refs else "normalized_excerpt"
            )
        evidence = EvidenceExcerpt(
            evidence_id=evidence_id,
            source_id=self.manifest.source_id,
            source_part_id=part_id,
            text_artifact_id=part.text_artifact_id,
            start_offset=start,
            end_offset=end,
            content_sha256=content_sha256,
            kind=part.kind,
            title=part.label[:300],
            content=content,
            image_refs=evidence_refs,
            locator=locator,
            resolved_version=self.manifest.resolved_version,
            extraction_method=extraction_method,
        )
        self.evidence[evidence_id] = evidence
        self.evidence_parts.add(part_id)
        self.evidence_parts.update(ref.source_part_id for ref in refs)
        return evidence

    def materialize_evidence_excerpts(self) -> list[EvidenceExcerpt]:
        """Create source-grounded text, image, or multimodal Evidence units."""
        self.evidence.clear()
        self.evidence_parts.clear()
        part_ids = self.curated_part_ids or [
            part_id
            for part_id, part in self.parts.items()
            if not part.metadata.get("container")
            and not part.metadata.get("layout_atom")
            and part.metadata.get("evidence_default", True)
        ]
        for part_id in part_ids:
            part = self.parts[part_id]
            if part.metadata.get("container") or part.metadata.get("evidence_default") is False:
                continue
            image_refs = [
                EvidenceImageRef.model_validate(item)
                for item in part.metadata.get("image_refs", [])
            ]
            if not part.text_artifact_id:
                if image_refs:
                    self.add_evidence_excerpt(part_id, 0, 0, image_refs)
                continue
            text = self.part_text.get(part_id, "")
            ranges = _excerpt_ranges(text, maximum=8_000)
            for index, (start, end) in enumerate(ranges):
                refs_for_range = [
                    ref
                    for ref in image_refs
                    if start <= ref.anchor_offset < end
                    or (index == len(ranges) - 1 and ref.anchor_offset >= end)
                ]
                self.add_evidence_excerpt(part_id, start, end, refs_for_range)
        return list(self.evidence.values())

    def evidence_candidates(self) -> list[SourcePart]:
        """Return parser-owned semantic candidates in stable source order."""
        return [
            part
            for part in self.parts.values()
            if part.part_id in self.normalized_parts
            and not part.metadata.get("container")
            and not part.metadata.get("layout_atom")
            and part.metadata.get("evidence_default", True)
            and not part.metadata.get("curated_projection")
            and not part.metadata.get("deterministic_projection")
        ]

    @staticmethod
    def _candidate_page_key(part: SourcePart) -> str:
        if part.locator.get("url"):
            return f"url:{part.locator['url']}"
        page = part.locator.get("page") or part.locator.get("page_start")
        if page is not None:
            return f"page:{page}"
        return f"parent:{part.parent_part_id or part.part_id}"

    def materialize_deterministic_evidence(
        self,
        *,
        target_chars: int = 4_000,
        maximum_chars: int = 10_000,
    ) -> list[EvidenceExcerpt]:
        """Pack adjacent source units into deterministic, page-local Evidence.

        Heading levels and source order decide boundaries. No semantic keyword
        scoring, model output, or rewritten source prose participates.
        """

        target_chars = max(1_000, min(int(target_chars), 8_000))
        maximum_chars = max(target_chars, min(int(maximum_chars), 12_000))
        candidates = self.evidence_candidates()
        pages: dict[str, list[SourcePart]] = {}
        for part in candidates:
            pages.setdefault(self._candidate_page_key(part), []).append(part)

        final_groups: list[list[SourcePart]] = []
        for page_parts in pages.values():
            page_parts.sort(key=lambda item: (item.ordinal, item.part_id))
            structural_groups: list[list[SourcePart]] = []
            current: list[SourcePart] = []
            for part in page_parts:
                level = int(part.locator.get("heading_level") or 0)
                starts_primary_section = level in {1, 2}
                if current and starts_primary_section:
                    structural_groups.append(current)
                    current = []
                current.append(part)
            if current:
                structural_groups.append(current)

            packed: list[list[SourcePart]] = []
            current = []
            current_chars = 0
            for group in structural_groups:
                group_chars = self._projection_char_count(group)
                if current and current_chars >= target_chars:
                    packed.append(current)
                    current = []
                    current_chars = 0
                if current and current_chars + group_chars > maximum_chars:
                    packed.append(current)
                    current = []
                    current_chars = 0
                current.extend(group)
                current_chars = self._projection_char_count(current)
            if current:
                if packed and self._projection_char_count(packed[-1] + current) <= maximum_chars:
                    packed[-1].extend(current)
                else:
                    packed.append(current)
            final_groups.extend(packed)

        projection_ids: list[str] = []
        group_records: list[dict[str, Any]] = []
        for group_index, group in enumerate(final_groups, start=1):
            projection = self._add_structural_projection(group, group_index)
            projection_ids.append(projection.part_id)
            group_records.append(
                {
                    "projection_part_id": projection.part_id,
                    "source_candidate_part_ids": [part.part_id for part in group],
                    "page_key": self._candidate_page_key(group[0]),
                    "char_count": projection.char_count,
                    "label": projection.label,
                }
            )

        self.curated_part_ids = projection_ids
        self.deterministic_evidence_plan = {
            "schema_version": "deterministic-evidence-plan.v1",
            "candidate_count": len(candidates),
            "page_count": len(pages),
            "group_count": len(final_groups),
            "target_chars": target_chars,
            "maximum_chars": maximum_chars,
            "groups": group_records,
        }
        plan_artifact = self.store.put_json_artifact(
            self.deterministic_evidence_plan,
            kind="deterministic_evidence_plan",
            source_id=self.manifest.source_id,
            preserve_text_integrity=True,
        )
        self.deterministic_evidence_plan["artifact_id"] = plan_artifact.artifact_id
        self.allowed_artifacts.add(plan_artifact.artifact_id)
        self.materialize_evidence_excerpts()
        self.persist()
        return list(self.evidence.values())

    def _projection_char_count(self, parts: list[SourcePart]) -> int:
        return sum(
            len(part.label) + 2 + len(self.part_text.get(part.part_id, ""))
            for part in parts
        ) + max(0, len(parts) - 1) * 2

    def _add_structural_projection(
        self, parts: list[SourcePart], group_index: int
    ) -> SourcePart:
        rendered: list[str] = []
        image_refs: list[dict[str, Any]] = []
        cursor = 0
        for part in parts:
            text = self.part_text.get(part.part_id, "")
            block = f"{part.label}\n\n{text}" if part.label else text
            if rendered:
                cursor += 2
            block_start = cursor
            rendered.append(block)
            for raw_ref in part.metadata.get("image_refs", []):
                adjusted = dict(raw_ref)
                adjusted["anchor_offset"] = (
                    block_start + len(part.label) + 2 + int(adjusted.get("anchor_offset", 0))
                )
                image_refs.append(adjusted)
            cursor += len(block)
        content = "\n\n".join(rendered)
        first, last = parts[0], parts[-1]
        part_id = _stable_id(
            "part",
            self.manifest.source_id,
            "deterministic-evidence",
            *(part.part_id for part in parts),
        )
        label = first.label if first.label == last.label else f"{first.label} — {last.label}"
        locator = dict(first.locator)
        locator.update(
            {
                "structured_from_part_ids": [part.part_id for part in parts],
                "structured_end_label": last.label,
                "structural_group_index": group_index,
            }
        )
        projection = SourcePart(
            part_id=part_id,
            source_id=self.manifest.source_id,
            kind=f"{self.manifest.primary_modality}_structured_section",
            label=label[:300],
            locator=locator,
            parent_part_id=first.parent_part_id,
            ordinal=first.ordinal,
            artifact_id=first.artifact_id,
            char_count=len(content),
            metadata={
                "deterministic_projection": True,
                "source_candidate_part_ids": [part.part_id for part in parts],
                "image_refs": image_refs,
            },
        )
        self.add_part(projection, content)
        self.inventoried_parts.add(projection.part_id)
        return projection

    def update_document_version(self, text: str) -> None:
        if self.manifest.source_metadata.get("discovery_candidate_id"):
            return
        detected = detect_document_version(text, self.target_profile.software_name)
        if detected == "unknown":
            return
        resolved, basis, conflict = resolve_version(detected, self.manifest.user_declared_version)
        self.manifest.document_declared_version = detected
        self.manifest.resolved_version = resolved
        self.manifest.version_resolution_basis = basis
        self.manifest.version_conflict = conflict
        if conflict:
            warning = (
                f"Source version {detected} conflicts with the user-declared version "
                f"{self.manifest.user_declared_version}; the source's own version takes precedence."
            )
            if warning not in self.warnings:
                self.warnings.append(warning)

    def coverage(self) -> Coverage:
        skipped_part_ids = {part_id for part_id in self.skipped if part_id in self.parts}
        return Coverage(
            discovered_parts=len(self.parts),
            inventoried_parts=len(self.inventoried_parts),
            normalized_parts=len(self.normalized_parts),
            semantic_units=sum(
                1
                for part_id, part in self.parts.items()
                if part_id in self.normalized_parts
                and not part.metadata.get("container")
                and not part.metadata.get("layout_atom")
                and not part.metadata.get("curated_projection")
                and not part.metadata.get("deterministic_projection")
            ),
            inspected_parts=len(self.inspected_parts),
            evidence_bearing_parts=len(self.evidence_parts),
            skipped_parts=len(skipped_part_ids),
            skipped=[{"part_id": key, "reason": value} for key, value in self.skipped.items()],
        )

    def persist(self) -> None:
        self._persist()

    def processable_part_ids(self) -> set[str]:
        return {
            part_id
            for part_id, part in self.parts.items()
            if not part.metadata.get("container") and not part.metadata.get("preserved_binary")
        }

    def build_corpus_artifact(self) -> str:
        # Evidence Architect runs materialize a compatibility EvidenceExcerpt
        # view from the accepted tree. Do not replace it with the legacy
        # one-SourcePart/one-Evidence projection here.
        if self.evidence_tree is None:
            self.materialize_evidence_excerpts()
        payload = {
            "schema_version": self.NORMALIZATION_VERSION,
            "source_id": self.manifest.source_id,
            "raw_artifact_id": self.manifest.raw_artifact_id,
            "manifest": self.manifest.model_dump(mode="json"),
            "parts": [part.model_dump(mode="json") for part in self.parts.values()],
            "evidence": [item.model_dump(mode="json") for item in self.evidence.values()],
            "evidence_tree": (
                self.evidence_tree.model_dump(mode="json")
                if self.evidence_tree is not None
                else None
            ),
            "coverage": self.coverage().model_dump(mode="json"),
            "gaps": self.gaps,
            "warnings": self.warnings,
        }
        artifact = self.store.put_json_artifact(
            payload,
            kind="normalized_source_corpus",
            source_id=self.manifest.source_id,
            metadata={"normalization_version": self.NORMALIZATION_VERSION},
            preserve_text_integrity=True,
        )
        self.corpus_artifact_id = artifact.artifact_id
        self.allowed_artifacts.add(artifact.artifact_id)
        self.store.write_source_json(
            self.manifest.source_id,
            "corpus.json",
            {"corpus_artifact_id": artifact.artifact_id, **payload},
        )
        return artifact.artifact_id

    def _persist(self) -> None:
        self.manifest.warnings = list(dict.fromkeys(self.warnings))
        self.store.write_source_json(
            self.manifest.source_id,
            "manifest.json",
            self.manifest.model_dump(mode="json"),
        )
        self.store.write_source_json(
            self.manifest.source_id,
            "parts.json",
            [part.model_dump(mode="json") for part in self.parts.values()],
        )
        self.store.write_source_json(
            self.manifest.source_id,
            "evidence.json",
            [item.model_dump(mode="json") for item in self.evidence.values()],
        )
        self.store.write_source_json(
            self.manifest.source_id,
            "workspace.json",
            {
                "inventory_completed": self.inventory_completed,
                "coverage": self.coverage().model_dump(mode="json"),
                "gaps": self.gaps,
                "warnings": self.warnings,
                "structure_submission": self.structure_submission,
                "curation_submission": self.curation_submission,
                "curated_part_ids": self.curated_part_ids,
                "deterministic_evidence_plan": self.deterministic_evidence_plan,
                "evidence_skeleton_plan_artifact_id": (
                    self.evidence_skeleton_plan_artifact_id
                ),
                "evidence_skeleton_tree_artifact_id": (
                    self.evidence_skeleton_tree_artifact_id
                ),
                "root_structure_artifact_id": self.root_structure_artifact_id,
                "pdf_document_structure_artifact_id": (
                    self.pdf_document_structure_artifact_id
                ),
                "evidence_tree_id": (
                    self.evidence_tree.evidence_tree_id
                    if self.evidence_tree is not None
                    else None
                ),
                "evidence_tree_artifact_id": self.evidence_tree_artifact_id,
                "corpus_artifact_id": self.corpus_artifact_id,
                "normalization_version": self.NORMALIZATION_VERSION,
            },
        )
