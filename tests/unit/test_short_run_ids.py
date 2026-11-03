import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from app.acquisition import AcquisitionService
from app.config import Settings
from app.models import SourceManifest, SourcePart, StagedInput, TargetProfile
from app.storage import RunStore
from app.toolkit import SourceWorkspace


class ShortRunIdTests(unittest.TestCase):
    def test_new_run_persists_canonical_short_ids_across_restarts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root, "run_short_ids")
            store.initialize({"target_profile": {
                "software_name": "Demo", "target_version": "1",
                "platform": "windows", "ui_locale": "en",
            }})
            self.assertEqual(store.canonical_id("E", "evi_long_one", "source", "one"), "E1")
            self.assertEqual(store.canonical_id("E", "evi_long_two", "source", "two"), "E2")
            self.assertEqual(store.canonical_id("T", "tsk_long_one", "task", "one"), "T1")
            resumed = RunStore(root, "run_short_ids")
            self.assertEqual(resumed.canonical_id("E", "evi_long_one", "source", "one"), "E1")
            self.assertEqual(resumed.canonical_id("E", "evi_long_three", "source", "three"), "E3")
            with closing(sqlite3.connect(resumed.root / "id_registry.sqlite3")) as db:
                rows = db.execute("SELECT namespace, fingerprint, ordinal FROM ids").fetchall()
            self.assertEqual(len(rows), 4)
            self.assertTrue(all("evi_long_one" not in str(row) for row in rows))

    def test_old_run_retains_existing_id_format(self) -> None:
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory), "run_legacy")
            (store.root / "state.json").write_text("{}", encoding="utf-8")
            self.assertEqual(store.canonical_id("E", "evi_original", "one"), "evi_original")
            self.assertFalse((store.root / "id_registry.sqlite3").exists())

    def test_evidence_is_short_in_the_persisted_source_workspace(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = TargetProfile(
                software_name="Demo", target_version="1", platform="windows", ui_locale="en"
            )
            store = RunStore(root, "run_short_source")
            store.initialize({"target_profile": target.model_dump(mode="json")})
            manifest = SourceManifest(
                source_id="SRC1", canonical_input_id="input_1", display_name="manual.txt",
                primary_modality="html", media_type="text/html",
            )
            workspace = SourceWorkspace(
                store, Settings(data_dir=root, output_dir=root / "output"),
                None, target, manifest,
            )
            workspace.add_part(
                SourcePart(
                    part_id="part_1", source_id=manifest.source_id,
                    kind="html_section", label="Step", locator={},
                ),
                "Click Add to create a graph.",
            )
            evidence = workspace.add_evidence_excerpt("part_1", 0, 28)
            self.assertEqual(evidence.evidence_id, "E1")
            self.assertEqual(
                workspace.add_evidence_excerpt("part_1", 0, 28).evidence_id, "E1"
            )

    def test_new_run_acquisition_assigns_short_source_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = TargetProfile(
                software_name="Demo", target_version="1", platform="windows", ui_locale="en"
            )
            store = RunStore(root, "run_short_acquisition")
            staged = store.stage_upload("IN1", "manual.html", b"<html><body>Graph instructions.</body></html>")
            store.initialize({"target_profile": target.model_dump(mode="json")})
            source = StagedInput(
                input_id="IN1", kind="uploaded_file", display_name="manual.html",
                staged_path=str(staged), declared_media_type="text/html",
            )
            settings = Settings(data_dir=root, output_dir=root / "output")
            result = AcquisitionService(store, settings).acquire([source], target)
            self.assertEqual(result.manifests[0].source_id, "SRC1")


if __name__ == "__main__":
    unittest.main()
