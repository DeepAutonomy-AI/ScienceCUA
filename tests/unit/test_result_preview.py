import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.knowledge import _stable_id
from app.result_preview import completed_child_skills, read_result_preview, read_source_previews
from app.storage import RunStore


def test_source_previews_show_manual_inputs_without_exposing_local_paths(tmp_path):
    store = RunStore(tmp_path, "run_source_preview")
    store.initialize({"target_profile": {"software_name": "Example", "target_version": "8.4.1", "platform": "General"}, "inputs": [{
        "input_id": "IN1", "kind": "uploaded_file", "display_name": "manual.pdf",
        "user_declared_version": "8.4.1", "staged_path": "C:/private/manual.pdf",
    }]})
    sources = read_source_previews(store, store.read_state())
    assert len(sources) == 1
    assert sources[0]["display_name"] == "manual.pdf"
    assert sources[0]["modality"] == "pdf"
    assert sources[0]["manifest"]["resolved_version"] == "8.4.1"
    assert "C:/private" not in json.dumps(sources)
    assert store.read_result() is None


def test_downloaded_source_metadata_replaces_the_selected_input_preview(tmp_path):
    store = RunStore(tmp_path, "run_source_preview")
    selected = store.put_json_artifact({"input": {
        "input_id": "IN1", "display_name": "Selected manual", "url": "https://example.org/manual",
    }}, kind="selected_source")
    manifest = {"source_id": "SRC1", "display_name": "Downloaded manual", "primary_modality": "html",
                "status": "failed", "origin_url": "https://example.org/manual", "resolved_version": "8.4.3",
                "source_origin": "websearch", "acquisition": {"headers": {"internal": "not public"}}}
    acquired = store.put_json_artifact({"manifests": [manifest]}, kind="acquisition_outcome")
    state = {"selected_source_artifact_id": selected.artifact_id, "acquisition_artifact_id": acquired.artifact_id}
    sources = read_source_previews(store, state)
    assert len(sources) == 1
    assert sources[0]["source_id"] == "SRC1"
    assert sources[0]["display_name"] == "Downloaded manual"
    assert sources[0]["status"] == "failed"
    assert "acquisition" not in sources[0]["manifest"]
    manifest.update(display_name="Updated manifest", status="excluded")
    store.write_source_json("SRC1", "manifest.json", manifest)
    sources = read_source_previews(store, state)
    assert len(sources) == 1
    assert sources[0]["display_name"] == "Updated manifest"
    assert sources[0]["status"] == "excluded"
    assert store.read_result() is None


class ResultPreviewTests(unittest.TestCase):
    def test_old_trajectory_preview_keeps_task_binding_and_drops_retired_fields(self):
        trajectory = {"trajectory_id": "TR1", "task_id": "T1", "goal": "Render a graph",
                      "steps": [], "citations": [], "reference_ids": ["R1"],
                      "reference_trigger_type": "required", "confidence": 0.9,
                      "caveats": ["Old note"], "unsupported_facets": ["Old missing feature"]}
        final = {"trajectories": [trajectory]}
        before = json.dumps(final, sort_keys=True)
        preview = read_result_preview(SimpleNamespace(read_result=lambda: final), {"phase": "completed"})
        self.assertEqual(preview["trajectories"][0]["task_id"], "T1")
        for field in ("goal", "reference_ids", "reference_trigger_type", "confidence", "caveats", "unsupported_facets"):
            self.assertNotIn(field, preview["trajectories"][0])
        self.assertEqual(json.dumps(final, sort_keys=True), before)

    def test_saved_profiles_and_task_mode_are_removed_from_preview_without_changing_artifacts(self):
        profile = {"software_name": "Example", "target_version": "1", "platform": "Linux",
                   "operation_path_preference": "gui", "operation_modes": ["GUI"]}
        task = {"task": "Render a graph", "category": "Display", "operation_mode": "gui", "citations": []}
        final = {"target_profile": profile, "snapshot": {"target_profile": profile},
                 "knowledge_snapshot": {"target_profile": profile, "tasks": [task]}, "tasks": [task]}
        before = json.dumps(final, sort_keys=True)
        preview = read_result_preview(SimpleNamespace(read_result=lambda: final), {"phase": "completed"})
        for container in (preview, preview["snapshot"], preview["knowledge_snapshot"]):
            self.assertNotIn("operation_path_preference", container["target_profile"])
            self.assertNotIn("operation_modes", container["target_profile"])
        self.assertNotIn("operation_mode", preview["tasks"][0])
        self.assertNotIn("operation_mode", preview["knowledge_snapshot"]["tasks"][0])
        self.assertEqual(json.dumps(final, sort_keys=True), before)

    def test_final_result_is_returned_without_reading_children(self):
        final = {"outcome": "succeeded", "skills": []}
        store = SimpleNamespace(read_result=lambda: final)
        self.assertIs(read_result_preview(store, {"phase": "completed"}), final)




    def test_placeholder_without_snapshot_is_unchanged(self):
        store = SimpleNamespace(read_result=lambda: None)
        self.assertIsNone(read_result_preview(store, {"phase": "registered"}))

    def test_historical_final_skills_drop_retired_fields_without_changing_saved_result(self):
        citation = {'evidence_id': 'E1', 'quote': 'Render the graph.'}
        final = {'skills': [{
            'reference_ids': ['R1'], 'reference_trigger_type': 'required',
            'task_ids': ['T1'], 'trajectory_ids': ['TR1'], 'caveats': ['Legacy caveat'],
            'confidence': 0.9, 'inferred': True, 'citations': [citation],
            'preconditions': [{'criterion': 'The graph is open.', 'citations': [citation]},
                              {'criterion': 'The graph is ready to render.'}],
            'procedure': [{'citations': [citation], 'expected_observation': 'Retired Skill observation'}],
            'completion_criteria': [{'criterion': 'The graph is rendered.', 'citations': [citation]}],
        }], 'references': [{'reference_id': 'R1'}]}
        before = json.dumps(final, sort_keys=True)
        preview = read_result_preview(SimpleNamespace(read_result=lambda: final), {'phase': 'completed'})
        for field in ('reference_ids', 'reference_trigger_type', 'task_ids', 'caveats',
                      'confidence', 'inferred', 'completion_criteria', 'citations'):
            self.assertNotIn(field, preview['skills'][0])
        self.assertEqual(preview['skills'][0]['trajectory_ids'], ['TR1'])
        self.assertEqual(preview['skills'][0]['preconditions'], 'The graph is open.\nThe graph is ready to render.')
        self.assertEqual(preview['skills'][0]['procedure'][0]['citations'], [citation])
        self.assertNotIn('expected_observation', preview['skills'][0]['procedure'][0])
        self.assertEqual(preview['references'], final['references'])
        self.assertEqual(json.dumps(final, sort_keys=True), before)

    def test_final_preview_drops_legacy_citation_rationale_globally_without_changing_saved_result(self):
        citation = {"evidence_id": "E1", "quote": "Render the graph.", "rationale": "Legacy explanation."}
        final = {
            "tasks": [{"citations": [citation], "confidence": 0.9}],
            "references": [{"citations": [citation], "confidence": 0.8,
                            "entries": [{"citations": [citation]}]}],
            "trajectories": [{"citations": [citation], "confidence": 0.7,
                              "steps": [{"citations": [citation]}]}],
            "skills": [{"preconditions": "", "procedure": [{"citations": [citation]}]}],
        }
        before = json.dumps(final, sort_keys=True)
        store = SimpleNamespace(read_result=lambda: final)
        preview = read_result_preview(store, {"phase": "completed"})
        for collection in ("tasks", "references", "trajectories"):
            self.assertNotIn("rationale", preview[collection][0]["citations"][0])
            if collection in {"tasks", "trajectories"}:
                self.assertNotIn("confidence", preview[collection][0])
            else:
                self.assertIn("confidence", preview[collection][0])
        for collection, nested in (("references", "entries"), ("trajectories", "steps"), ("skills", "procedure")):
            self.assertNotIn("rationale", preview[collection][0][nested][0]["citations"][0])
        self.assertEqual(json.dumps(final, sort_keys=True), before)

    def test_only_accepted_proposals_from_completed_matching_children_are_visible(self):
        citation = {"evidence_id": "evi_one", "quote": "Render the graph", "rationale": "Legacy explanation."}
        def draft(title):
            return {"title": title, "description": "Render the graph",
                "applicability": "When a graph is ready", "task_ids": ["tsk_one"],
                "trajectory_ids": ["traj_one"], "citations": [citation],
                "reference_ids": ["ref_one"], "reference_trigger_type": "required",
                "caveats": [], "confidence": 0.9, "inferred": True, "completion_criteria": [],
                "preconditions": [{"criterion": "The graph is open.", "citations": [citation]}],
                "procedure": [{"ordinal": 1, "instruction": "Render the graph",
                    "expected_observation": "Graph rendered", "citations": [citation]}]}
        attempts = [SimpleNamespace(stage="skill_induction", status=status,
            attempt_id=name, capsule_artifact_id=name) for name, status in
            (("done", "succeeded"), ("open", "running"), ("failed", "failed"), ("wrong", "succeeded"))]
        events = []
        for attempt, title, accepted in (("done", "Rejected", False), ("done", "Accepted", True),
                ("open", "Unfinished", True), ("failed", "Failed child", True), ("wrong", "Other inputs", True)):
            events.append({"event_type": "capability_tool_call", "timestamp": "2026-09-17T00:00:00Z",
                "payload": {"tool": "submit_skill_batch", "attempt_id": attempt,
                    "arguments": {"skills": [draft(title)]}}})
            events.append({"event_type": "knowledge_tool_observation", "payload": {
                "stage": "skill_induction", "tool": "submit_skill_batch", "observation": {
                    "candidate_ids": [_stable_id("skl", "knowledge_one", title.lower(), "ref_one", "traj_one")] if accepted else []}}})
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            audit.write_text("\n".join(json.dumps(item) for item in events) + "\n{", encoding="utf-8")
            before = audit.read_bytes()
            store = SimpleNamespace(audit_path=audit, list_attempts=lambda: attempts,
                read_json_artifact=lambda name: {"input_artifact_ids": ["corpus", "knowledge" if name != "wrong" else "other"],
                    "scope": {"trajectory_ids": ["traj_one"]}})
            state = {"corpus_snapshot_artifact_id": "corpus", "knowledge_snapshot_artifact_id": "knowledge"}
            knowledge = SimpleNamespace(knowledge_snapshot_id="knowledge_one", tasks=[], references=[])
            with patch("app.result_preview.validate_skills", side_effect=lambda skills, *args: (skills, [])):
                skills, done, total = completed_child_skills(store, state, knowledge, SimpleNamespace(evidence=[]))
            self.assertEqual([item["title"] for item in skills], ["Accepted"])
            self.assertNotIn("citations", skills[0])
            self.assertNotIn("reference_ids", skills[0])
            self.assertNotIn("reference_trigger_type", skills[0])
            for field in ("task_ids", "caveats", "confidence", "inferred", "completion_criteria"):
                self.assertNotIn(field, skills[0])
            self.assertEqual(skills[0]["preconditions"], "The graph is open.")
            self.assertEqual(skills[0]["procedure"][0]["citations"][0]["evidence_id"], citation["evidence_id"])
            self.assertEqual(skills[0]["procedure"][0]["citations"][0]["quote"], citation["quote"])
            self.assertNotIn("rationale", skills[0]["procedure"][0]["citations"][0])
            self.assertNotIn("expected_observation", skills[0]["procedure"][0])
            self.assertEqual((done, total), (1, 4))
            self.assertEqual(audit.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
