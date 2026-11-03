from __future__ import annotations

import hashlib

from app.models import EvidenceExcerpt, EvidenceNode, EvidenceSourceSpan, EvidenceTree


def first_level_tree(evidence: EvidenceExcerpt, chapter: str = "Operations") -> EvidenceTree:
    digest = hashlib.sha256(evidence.content.encode()).hexdigest()
    branch = EvidenceNode(
        evidence_node_id="N2", source_id=evidence.source_id, node_key="chapter",
        parent_node_id="N1", ordinal=1, node_type="section", title=chapter,
        title_origin="source", source_refs=[EvidenceSourceSpan(
            source_part_id=evidence.source_part_id, text_artifact_id=evidence.text_artifact_id,
            start_offset=evidence.start_offset, end_offset=evidence.end_offset,
            content_sha256=digest,
        )], content=evidence.content, content_artifact_id=evidence.text_artifact_id,
        content_sha256=digest,
    )
    root = EvidenceNode(
        evidence_node_id="N1", source_id=evidence.source_id, node_key="root",
        child_node_ids=["N2"], ordinal=0, node_type="document",
        title="Manual", title_origin="source",
    )
    return EvidenceTree(evidence_tree_id="ET1", source_id=evidence.source_id,
                        root_node_id="N1", nodes=[root, branch],
                        plan_artifact_id="plan", validation_artifact_id="validation")
