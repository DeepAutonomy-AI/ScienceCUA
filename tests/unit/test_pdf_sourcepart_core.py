"""Portable PDF extraction checks; fixtures are authored in temporary storage."""
from __future__ import annotations

import hashlib
import errno
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import pymupdf

from app.pdf_sourcepart import extractor, fallback


class PDFCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pdf-core-")
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "manual.pdf"
        cls.locked = cls.root / "locked.pdf"
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=800)
            page.insert_text((45, 60), "Methods", fontsize=22, fontname="hebo")
            page.insert_text((45, 100), "PARAGRAPH_BEGIN is a wrapped paragraph.\nIts continuation ends with PARAGRAPH_END.", fontsize=11)
            page.insert_text((45, 155), "- LIST_FIRST\n- LIST_SECOND\n  A continuation belongs to the second list item.", fontsize=11)
            for y in (220, 250, 280, 310):
                page.draw_line((45, y), (530, y))
            for x in (45, 180, 530):
                page.draw_line((x, 220), (x, 310))
            for y, cells in ((240, ("Key", "Action")), (270, ("K1", "ACTION_ONE")), (300, ("K2", "ACTION_TWO"))):
                page.insert_text((55, y), cells[0], fontsize=11)
                page.insert_text((190, y), cells[1], fontsize=11)
            page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(45, 84, 200, 102), "uri": "https://example.org/reference#table"})
            page = document.new_page(width=600, height=800)
            page.insert_text((45, 60), "Commands", fontsize=22, fontname="hebo")
            left = "LEFT_COLUMN_FIRST explains the first operation.\nThis column describes its own series of actions.\nLEFT_COLUMN_LAST ends this discussion."
            right = "RIGHT_COLUMN_FIRST starts a separate subject.\nThis side explains a different operation in prose.\nRIGHT_COLUMN_LAST closes the second column."
            page.insert_text((45, 110), left, fontsize=10)
            page.insert_text((320, 110), right, fontsize=10)
            page.insert_text((45, 210), "code_alpha(x)\ncode_beta(y)", fontsize=11, fontname="cour")
            page = document.new_page(width=600, height=800)
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
            pixmap.clear_with(120)
            page.insert_image(pymupdf.Rect(80, 90, 280, 290), pixmap=pixmap)
            document.new_page(width=600, height=800)
            document.set_toc([[1, "Methods", 1], [1, "Commands", 2]])
            document.save(cls.source)
            document.save(cls.locked, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="fixture-owner", user_pw="fixture-open")
        cls.document = extractor.extract_pdf(cls.source, cls.root / "normal", source_id="source-test")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def text(self, document=None):
        return "\n".join(part["content"] for part in (document or self.document)["parts"])

    def test_prose_lists_code_and_links_survive(self):
        text = self.text()
        for value in ("PARAGRAPH_BEGIN", "PARAGRAPH_END", "LIST_FIRST", "LIST_SECOND", "code_alpha(x)", "code_beta(y)"):
            self.assertIn(value, text)
        paragraphs = [b for p in self.document["pages"] for b in p["blocks"] if b["type"] == "paragraph"]
        self.assertTrue(any("PARAGRAPH_BEGIN" in b["text"] and "PARAGRAPH_END" in b["text"] for b in paragraphs))
        self.assertIn("https://example.org/reference#table", json.dumps(self.document["links"]))

    def test_table_rows_keep_their_cell_relationships(self):
        tables = [b for p in self.document["pages"] for b in p["blocks"] if b["type"] == "table"]
        self.assertTrue(tables)
        rows = [[extractor.plain(cell["runs"]) for cell in row["cells"]] for table in tables for row in table["rows"]]
        self.assertIn(["K1", "ACTION_ONE"], rows)
        self.assertIn(["K2", "ACTION_TWO"], rows)

    def test_columns_read_top_to_bottom_before_switching_column(self):
        text = self.text()
        self.assertLess(text.index("LEFT_COLUMN_FIRST"), text.index("LEFT_COLUMN_LAST"))
        self.assertLess(text.index("LEFT_COLUMN_LAST"), text.index("RIGHT_COLUMN_FIRST"))
        self.assertLess(text.index("RIGHT_COLUMN_FIRST"), text.index("RIGHT_COLUMN_LAST"))

    def test_image_and_blank_pages_remain_represented(self):
        pages = self.document["pages"]
        self.assertTrue(any(block["type"] == "image" for block in pages[2]["blocks"]))
        self.assertEqual(pages[3]["content_kind"], "blank_page")
        self.assertEqual(self.document["summary"]["pages"], 4)
        self.assertTrue(self.document["summary"]["all_pages_represented"])
        for image in self.document["images"]:
            self.assertEqual(image["status"], "saved")
            self.assertTrue((self.root / "normal" / "assets" / image["filename"]).is_file())

    def test_source_bytes_line_coverage_and_offsets_are_auditable(self):
        self.assertEqual((self.root / "normal" / "raw" / "source.pdf").read_bytes(), self.source.read_bytes())
        self.assertTrue(self.document["summary"]["structured_line_coverage"])
        for part in self.document["parts"]:
            self.assertEqual(part["char_count"], len(part["content"]))
            self.assertEqual(part["content_sha256"], hashlib.sha256(part["content"].encode()).hexdigest())
            ranges = part["metadata"]["block_ranges"]
            blocks = [b for b in part["metadata"]["blocks"] if extractor.block_markdown(b)]
            for span, block in zip(ranges, blocks, strict=True):
                self.assertEqual(part["content"][span["start_offset"]:span["end_offset"]], extractor.block_markdown(block))

    def test_soft_budget_keeps_one_oversized_code_block_whole(self):
        body = "\n".join(f"CODE_TOKEN_{i:03d} = " + "x" * 130 for i in range(95))
        page = {"page": 1, "mode": "structured", "blocks": [{"type": "code", "text": body, "page": 1, "line_ids": []}]}
        parts = extractor.make_parts([page], "source-code", 10_000)
        self.assertEqual(len(parts), 1)
        self.assertIn(body, parts[0]["content"])
        self.assertTrue(parts[0]["metadata"]["soft_limit_exceeded"])

    def test_forced_page_failure_retains_text_visual_and_other_pages(self):
        document = extractor.extract_pdf(self.source, self.root / "forced", fail_pages={1})
        self.assertEqual(document["pages"][0]["mode"], "basic_text")
        self.assertIn("PARAGRAPH_BEGIN", self.text(document))
        self.assertEqual(document["pages"][1]["mode"], "structured")
        self.assertTrue(document["pages"][0]["assets"])
        self.assertTrue(document["summary"]["all_pages_represented"])

    def test_correct_password_reads_content(self):
        document = extractor.extract_pdf(self.locked, self.root / "unlocked", password="fixture-open")
        self.assertEqual(document["summary"]["pages"], 4)
        self.assertIn("PARAGRAPH_BEGIN", self.text(document))

    def test_locked_pdf_keeps_original_without_inventing_page_count(self):
        document = extractor.extract_pdf(self.locked, self.root / "locked-result")
        self.assertIsNone(document["summary"]["pages"])
        self.assertIsNone(document["summary"]["all_pages_represented"])
        self.assertTrue(document["parts"])
        self.assertEqual((self.root / "locked-result" / "raw" / "source.pdf").read_bytes(), self.locked.read_bytes())

    def test_damaged_pdf_keeps_original_and_explicit_gap(self):
        source = self.root / "damaged.pdf"
        source.write_bytes(b"%PDF-1.7\ntruncated")
        document = extractor.extract_pdf(source, self.root / "damaged-result")
        self.assertTrue(document["parts"])
        self.assertEqual(document["pages"][0]["mode"], "raw_pdf")
        self.assertIsNone(document["summary"]["pages"])
        self.assertEqual((self.root / "damaged-result" / "raw" / "source.pdf").read_bytes(), source.read_bytes())

    def test_independent_engine_recovers_every_page(self):
        with patch.object(extractor, "fitz", None):
            document = extractor.extract_pdf(self.source, self.root / "independent")
        self.assertEqual([p["page"] for p in document["pages"]], [1, 2, 3, 4])
        self.assertIn("PARAGRAPH_BEGIN", self.text(document))
        self.assertIn("RIGHT_COLUMN_LAST", self.text(document))
        self.assertTrue(all(p["mode"] != "structured" for p in document["pages"]))

    def test_document_closes_on_output_error(self):
        document = pymupdf.open(self.source)
        with patch.object(extractor.fitz, "open", return_value=document), patch.object(extractor, "save_json", side_effect=PermissionError("output unavailable")):
            with self.assertRaises(PermissionError):
                extractor.extract_pdf(self.source, self.root / "write-fails")
        self.assertTrue(document.is_closed)

    def test_native_outline_is_never_read_and_page_bodies_survive(self):
        with patch.object(pymupdf.Document, "get_toc", side_effect=AssertionError("native bookmarks must not be read")) as get_toc:
            document = extractor.extract_pdf(self.source, self.root / "bad-outline")
        self.assertIn("PARAGRAPH_BEGIN", self.text(document))
        self.assertEqual(document["outline"], [])
        self.assertFalse(any(item['code'] == 'outline_unavailable' for item in document['warnings']))
        get_toc.assert_not_called()

    def test_failed_document_inference_recovers_every_page(self):
        with patch.object(extractor, "document_structure", side_effect=ValueError("unfamiliar metadata")):
            document = extractor.extract_pdf(self.source, self.root / "bad-structure")
        self.assertEqual(document["summary"]["pages"], 4)
        self.assertTrue(all(p["mode"] != "structured" for p in document["pages"]))
        self.assertIn("PARAGRAPH_BEGIN", self.text(document))
        self.assertIn("RIGHT_COLUMN_LAST", self.text(document))
        self.assertEqual(document["warnings"][0]["code"], "document_structure_unavailable")

    def test_narrow_key_value_columns_remain_a_table(self):
        lines = []
        for row in range(4):
            for col, (start, end, text) in enumerate([(10, 24, f"K{row}"), (37, 115, f"Action {row}")]):
                bbox = [start, row * 12, end, row * 12 + 10]
                lines.append({"page": 1, "line_id": f"{row}:{col}", "bbox": bbox, "size": 10,
                              "text": text, "runs": [{"text": text, "bbox": bbox, "size": 10, "flags": 0}]})
        tables, owned = extractor.infer_tables(lines, {})
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(owned), 8)
        self.assertEqual([[extractor.plain(c["runs"]) for c in row["cells"]] for row in tables[0]["rows"]],
                         [[f"K{i}", f"Action {i}"] for i in range(4)])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pdf-recovery-")
        self.output = Path(self.temporary.name)
        self.pdf = self.output / "用户手册.pdf"

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, **kwargs):
        return fallback.recover_page(self.pdf, 2, output_dir=self.output, **kwargs)

    def test_plain_text_preserves_full_source_text(self):
        page = Mock()
        text = "标题\n\n第一段。\n换行。\n\n代码 >>> f(x)\n"
        page.get_text.return_value = text
        result = self.call(pymupdf_page=page)
        self.assertEqual(result["text"], text)
        self.assertEqual(result["mode"], "basic_text")
        self.assertEqual(len(result["blocks"]), 3)
        page.get_text.assert_called_once_with("text", sort=True)

    def test_pypdf_is_independent_second_engine(self):
        page = Mock()
        for error in (RuntimeError, OSError):
            with self.subTest(error=error.__name__):
                page.get_text.side_effect = error("secret-password")
                with patch.object(fallback, "_pypdf_text", return_value="recovered\n"):
                    result = self.call(pymupdf_page=page, reason="sk-secret")
                self.assertEqual(result["recovery_engine"], "pypdf")
                self.assertNotIn("secret", str(result["warnings"]))

    def test_poppler_text_is_third_engine(self):
        with patch.object(fallback, "_pypdf_text", side_effect=ValueError), patch.object(fallback, "_poppler_text", return_value="layout text"):
            result = self.call()
        self.assertEqual(result["recovery_engine"], "poppler_pdftotext")

    @staticmethod
    def write_png(*args):
        target = args[3]
        target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (20).to_bytes(4, "big") + (30).to_bytes(4, "big"))
        return True

    def test_no_machine_text_returns_explicit_page_image(self):
        with patch.object(fallback, "_pypdf_text", return_value=""), patch.object(fallback, "_poppler_text", return_value=""), patch.object(fallback, "_render_poppler", side_effect=self.write_png):
            result = self.call()
        self.assertEqual(result["mode"], "page_image")
        self.assertEqual(result["capture_status"], "page_visual_preserved_no_machine_text")
        self.assertEqual(result["assets"][0]["width"], 20)
        self.assertEqual(result["assets"][0]["height"], 30)
        self.assertFalse(Path(result["assets"][0]["path"]).is_absolute())

    def test_all_parse_and_render_failures_retain_original(self):
        expected = {"mode": "raw_pdf", "capture_status": "original_preserved_page_unreadable"}
        with patch.object(fallback, "_pypdf_text", side_effect=ValueError), patch.object(fallback, "_poppler_text", return_value=None), patch.object(fallback, "_render_poppler", side_effect=subprocess.TimeoutExpired("pdftoppm", 1)), patch.object(fallback, "_render_pymupdf", return_value=False), patch.object(fallback, "_raw_result", return_value=expected) as preserve:
            result = self.call()
        self.assertIs(result, expected)
        preserve.assert_called_once()
        self.assertIn("poppler_render_failed:TimeoutExpired", preserve.call_args.args[3])

    def test_password_is_not_passed_to_external_processes(self):
        with patch.object(fallback, "_pypdf_text", return_value=""), patch.object(fallback, "_poppler_text") as external_text, patch.object(fallback, "_render_poppler") as external_render, patch.object(fallback, "_render_pymupdf", return_value=False), patch.object(fallback, "_raw_result", return_value={"mode": "raw_pdf"}):
            self.call(password="private")
        external_text.assert_not_called()
        external_render.assert_not_called()

    def test_cancellation_is_never_swallowed(self):
        page = Mock()
        page.get_text.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.call(pymupdf_page=page)

    def test_storage_errors_are_never_reported_as_capture(self):
        with patch.object(fallback, "_pypdf_text", return_value=""), patch.object(fallback, "_poppler_text", return_value=""), patch.object(fallback, "_render_poppler", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                self.call()

    def test_native_render_decode_oserror_uses_independent_renderer(self):
        page = Mock()
        page.rect = pymupdf.Rect(0, 0, 100, 100)
        page.get_pixmap.side_effect = OSError("unsupported rendering stream")
        with patch.object(fallback, "_render_poppler", side_effect=self.write_png):
            asset = fallback.preserve_page_visual(self.pdf, 0, pymupdf_page=page, output_dir=self.output)
        self.assertIsNotNone(asset)
        self.assertEqual(asset["recovery_engine"], "poppler_pdftoppm")

    def test_native_render_output_write_errors_propagate(self):
        page = Mock()
        page.rect = pymupdf.Rect(0, 0, 100, 100)
        pixmap = page.get_pixmap.return_value
        pixmap.width = pixmap.height = 100
        pixmap.tobytes.return_value = b"rendered PNG bytes"
        for error in (PermissionError("unwritable output"), OSError(errno.ENOSPC, "full disk"), OSError("write failure")):
            with self.subTest(error=type(error).__name__), patch.object(Path, "write_bytes", side_effect=error):
                with self.assertRaises(type(error)):
                    fallback._render_pymupdf(page, self.output / "page.png", 4_000_000)

    def test_invalid_png_uses_original_document_recovery(self):
        with patch.object(fallback, "_pypdf_text", return_value=""), patch.object(fallback, "_poppler_text", return_value=""), patch.object(fallback, "_render_poppler", return_value=True), patch.object(fallback, "_image_asset", side_effect=ValueError), patch.object(fallback, "_raw_result", return_value={"mode": "raw_pdf"}) as preserve:
            result = self.call()
        self.assertEqual(result["mode"], "raw_pdf")
        self.assertIn("rendered_image_invalid:ValueError", preserve.call_args.args[3])

    def test_invalid_budgets_are_rejected(self):
        for kwargs in ({"max_render_pixels": 0}, {"timeout": float("nan")}, {"timeout": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.call(**kwargs)

    def test_severe_replacement_text_tries_independent_engine(self):
        page = Mock()
        page.get_text.return_value = "\ufffd\ufffd\x00\x01正文\n" + " " * 100
        with patch.object(fallback, "_pypdf_text", return_value="清楚的正文 ∑x² = α\n"):
            result = self.call(pymupdf_page=page)
        self.assertEqual(result["recovery_engine"], "pypdf")
        self.assertIn("pymupdf_text_unreadable_glyphs", result["warnings"])

    def test_all_unreadable_text_engines_use_visual_preservation(self):
        page = Mock()
        page.get_text.return_value = "\ufffd\ufffdA"
        page.rect = [0, 0, 100, 100]
        with patch.object(fallback, "_pypdf_text", return_value="\x00\x00A"), patch.object(fallback, "_poppler_text", return_value="\x7f\x80A"), patch.object(fallback, "_render_poppler", side_effect=self.write_png):
            result = self.call(pymupdf_page=page)
        self.assertEqual(result["mode"], "page_image")
        for engine in ("pymupdf", "pypdf", "poppler"):
            self.assertIn(f"{engine}_text_unreadable_glyphs", result["warnings"])

    def test_non_ascii_math_and_private_use_text_are_not_rejected(self):
        values = ["中文章节与段落", "∑ α β ∫ ∞ ⊆ × ² √", "français naïve façade", "مرحبا بالعالم", "\ue001\ue002", "meaningful text \ufffd"]
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(fallback.text_quality(value)["usable"])

    def test_exact_half_invalid_is_unreadable_and_whitespace_cannot_hide_it(self):
        for value in ("X\ufffd", "X\x00\n" + " " * 200, "X\ud800"):
            with self.subTest(value=repr(value)):
                quality = fallback.text_quality(value)
                self.assertFalse(quality["usable"])
                self.assertEqual(quality["invalid_character_ratio"], 0.5)

    def test_basic_text_can_keep_full_page_visual_without_losing_text(self):
        page = Mock()
        page.get_text.return_value = "Readable page header\n"
        visual = {"id": "page-image-0003", "path": "assets/page-3.png", "bbox": [0, 0, 100, 100]}
        with patch.object(fallback, "preserve_page_visual", return_value=visual) as preserve:
            result = self.call(pymupdf_page=page, include_visual=True)
        self.assertEqual(result["mode"], "basic_text")
        self.assertEqual(result["text"], "Readable page header\n")
        self.assertEqual(result["assets"], [visual])
        self.assertEqual(result["capture_status"], "machine_text_recovered")
        self.assertNotIn("unavailable", result["blocks"][-1]["text"])
        preserve.assert_called_once()

    def test_visual_preservation_tries_next_renderer_independently(self):
        warnings = []
        with patch.object(fallback, "_render_pymupdf", side_effect=RuntimeError), patch.object(fallback, "_render_poppler", side_effect=self.write_png):
            asset = fallback.preserve_page_visual(self.pdf, 2, output_dir=self.output, warnings=warnings)
        self.assertEqual(asset["role"], "fallback_full_page_visual")
        self.assertEqual(asset["recovery_engine"], "poppler_pdftoppm")
        self.assertIn("visual_pymupdf_pixmap_failed:RuntimeError", warnings)



