"""Independent PDF recovery; no dependence on the structured extractor.

The recovery ladder preserves machine text when available, otherwise a bounded
page image, otherwise the complete original PDF.  A page image is deliberately
not presented as OCR text.  Storage errors and cancellation remain visible.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import unicodedata

from .filesystem import filesystem_path


DEFAULT_RENDER_PIXELS = 4_000_000
DEFAULT_TOOL_TIMEOUT = 20.0


class PDFDecodeError(RuntimeError):
    """A decoder failed without reporting an operating-system storage error."""


def decode_call(function, *args, **kwargs):
    """Separate pure parser/renderer failures from source/output file errors.

    Some native decoders report unsupported streams as an errno-less OSError.
    Only decoder calls belong inside this boundary: output writes remain outside.
    PermissionError and errors carrying an OS errno must still reach the caller.
    """
    try:
        return function(*args, **kwargs)
    except OSError as exc:
        if isinstance(exc, PermissionError) or exc.errno is not None:
            raise
        raise PDFDecodeError("PDF decoder failed") from exc


def _code(prefix: str, exc: Exception | None = None) -> str:
    # Exception messages may contain PDF text, filenames or credentials.  The
    # stable operation and exception type identify failure without exposing it.
    return prefix if exc is None else f"{prefix}:{type(exc).__name__}"


def text_quality(value: str) -> dict[str, Any]:
    """Detect severely broken glyph decoding without guessing a language.

    CJK, accents, mathematical symbols and private-use glyphs are allowed.
    Replacement characters, illegal controls and unpaired surrogates are the
    only rejected character classes.  Whitespace cannot dilute their ratio.
    """
    invalid = 0
    meaningful = 0
    for character in value:
        broken = (
            character == "\ufffd"
            or unicodedata.category(character) == "Cs"
            or (unicodedata.category(character) == "Cc" and character not in "\t\r\n")
        )
        if broken:
            invalid += 1
        elif not character.isspace():
            meaningful += 1
    inspected = invalid + meaningful
    ratio = invalid / inspected if inspected else 0.0
    unreadable = bool(invalid and ratio >= 0.5)
    return {
        "usable": bool(meaningful and not unreadable),
        "unreadable_glyphs": unreadable,
        "invalid_character_count": invalid,
        "inspected_character_count": inspected,
        "invalid_character_ratio": ratio,
    }


def _has_text(value: str) -> bool:
    return text_quality(value)["usable"]


def _usable_text(value: str | None, engine: str, warnings: list[str]) -> bool:
    quality = text_quality(value) if isinstance(value, str) else {"usable": False, "unreadable_glyphs": False}
    if quality["usable"]:
        return True
    warnings.append(f"{engine}_text_unreadable_glyphs" if quality["unreadable_glyphs"] else f"{engine}_text_unavailable_or_empty")
    return False


def _bbox(page: Any) -> list[float] | None:
    try:
        values = [float(value) for value in page.rect]
        if len(values) == 4 and all(math.isfinite(value) for value in values):
            return values
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return None


def _paragraphs(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(r"(?:[^\n]|\n(?![ \t]*\n))+", text):
        if match.group().strip():
            blocks.append({
                "type": "paragraph", "text": match.group(), "bbox": None,
                "line_ids": [], "text_start": match.start(),
                "text_end": match.end(), "recovered": True,
            })
    return blocks


def _text_result(text: str, engine: str, warnings: list[str], page_index: int) -> dict[str, Any]:
    return {
        "mode": "basic_text", "text": text, "blocks": _paragraphs(text),
        "assets": [], "warnings": warnings, "degraded": True,
        "capture_status": "machine_text_recovered", "recovery_engine": engine,
        "page_index": page_index,
    }


def _find_tool(name: str, poppler_dir: Path | None = None) -> str | None:
    if poppler_dir:
        for filename in (name + ".exe", name):
            candidate = Path(poppler_dir) / filename
            if candidate.is_file():
                return str(candidate)
    located = shutil.which(name)
    if located:
        return located
    # Some runtime distributions expose only pdftoppm on PATH.
    renderer = shutil.which("pdftoppm")
    if renderer:
        for filename in (name + ".exe", name):
            candidate = Path(renderer).parent / filename
            if candidate.is_file():
                return str(candidate)
    return None


def _run_tool(arguments: list[str], working_dir: Path, timeout: float) -> int:
    # File-backed output avoids an unbounded stderr/stdout allocation.  Never
    # include these streams in error reports: they may contain source data.
    with tempfile.TemporaryFile(dir=working_dir) as stream:
        completed = subprocess.run(
            arguments, stdout=stream, stderr=stream, timeout=timeout,
            check=False, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return completed.returncode


def _pypdf_text(pdf_path: Path, page_index: int, password: str | None) -> str:
    module = importlib.import_module("pypdf")
    reader = module.PdfReader(str(pdf_path), strict=False)
    try:
        if reader.is_encrypted and not reader.decrypt(password or ""):
            raise ValueError("PDF password unavailable")
        return reader.pages[page_index].extract_text() or ""
    finally:
        close = getattr(reader, "close", None)
        if close:
            close()


def _poppler_text(
    pdf_path: Path, page_index: int, output_dir: Path,
    timeout: float, poppler_dir: Path | None,
) -> str | None:
    executable = _find_tool("pdftotext", poppler_dir)
    if not executable:
        return None
    with tempfile.TemporaryDirectory(prefix="pdf-text-", dir=output_dir) as temp:
        target = Path(temp) / "page.txt"
        status = _run_tool([
            executable, "-f", str(page_index + 1), "-l", str(page_index + 1),
            "-layout", "-enc", "UTF-8", str(pdf_path), str(target),
        ], Path(temp), timeout)
        if status != 0 or not target.is_file():
            raise RuntimeError("pdftotext failed")
        # A malformed document must not make a subprocess produce arbitrarily
        # large text.  The next recovery level retains the original source.
        if target.stat().st_size > 32_000_000:
            raise ValueError("recovered text exceeds process output budget")
        return target.read_text(encoding="utf-8", errors="replace").rstrip("\f")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_asset(path: Path, output_dir: Path, page_index: int, bbox: list[float] | None) -> dict[str, Any]:
    # A PNG header carries dimensions without decoding an entire image.
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24:
        raise ValueError("renderer returned a non-PNG image")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return {
        "id": f"page-image-{page_index + 1:04d}",
        "image_id": f"page-image-{page_index + 1:04d}",
        "path": path.relative_to(output_dir).as_posix(),
        "media_type": "image/png", "content_type": "image/png",
        "sha256": _hash_file(path), "size_bytes": path.stat().st_size,
        "width": width, "height": height, "bbox": bbox,
        "page_index": page_index, "role": "fallback_page_image",
    }


def _render_poppler(
    pdf_path: Path, page_index: int, output_dir: Path, target: Path,
    timeout: float, max_render_pixels: int, poppler_dir: Path | None,
) -> bool:
    executable = _find_tool("pdftoppm", poppler_dir)
    if not executable:
        return False
    with tempfile.TemporaryDirectory(prefix="pdf-render-", dir=output_dir) as temp:
        prefix = Path(temp) / "page"
        # Scaling the longest side to sqrt(budget) bounds both sides, including
        # a maliciously oversized MediaBox.  Unlike -r this is a pixel bound.
        longest_side = max(1, int(math.sqrt(max_render_pixels)))
        status = _run_tool([
            executable, "-f", str(page_index + 1), "-l", str(page_index + 1),
            "-singlefile", "-scale-to", str(longest_side), "-png",
            str(pdf_path), str(prefix),
        ], Path(temp), timeout)
        produced = prefix.with_suffix(".png")
        if status != 0 or not produced.is_file():
            raise RuntimeError("pdftoppm failed")
        # Destination write failures are not extraction failures.
        shutil.copyfile(produced, target)
    return True


def _render_pymupdf(page: Any, target: Path, max_render_pixels: int) -> bool:
    if page is None:
        return False
    values = _bbox(page)
    if values is None:
        raise ValueError("page geometry unavailable")
    width, height = values[2] - values[0], values[3] - values[1]
    if width <= 0 or height <= 0:
        raise ValueError("invalid page geometry")
    module = importlib.import_module("pymupdf")
    # A little margin accounts for pixel rounding in MuPDF.
    factor = min(1.5, math.sqrt(max_render_pixels / (width * height)) * 0.98)
    pixmap = decode_call(page.get_pixmap, matrix=module.Matrix(factor, factor), alpha=False)
    if pixmap.width * pixmap.height > max_render_pixels:
        raise ValueError("renderer exceeded pixel budget")
    # File writes are deliberately outside the caller's recovery exception
    # handling.  A full/unwritable disk cannot be reported as captured content.
    encoded = decode_call(pixmap.tobytes, "png")
    target.write_bytes(encoded)
    return True


def preserve_page_visual(
    pdf_path: Path, page_index: int, *, pymupdf_page: Any = None,
    password: str | None = None, output_dir: Path,
    max_render_pixels: int = DEFAULT_RENDER_PIXELS,
    timeout: float = DEFAULT_TOOL_TIMEOUT, poppler_dir: Path | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any] | None:
    """Keep the entire page's visual content beside a text recovery.

    No layout parsing or OCR is involved.  An authenticated page can render
    directly; external tools are skipped when a password was supplied.  This
    optional derivative does not replace the original PDF or machine text.
    """
    if page_index < 0:
        raise ValueError("page_index must be nonnegative")
    if not isinstance(max_render_pixels, int) or max_render_pixels < 1:
        raise ValueError("max_render_pixels must be a positive integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    pdf_path, output_dir = filesystem_path(pdf_path), filesystem_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    warnings = warnings if warnings is not None else []
    path_digest = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:12]
    target = output_dir / "assets" / f"page-{page_index + 1:04d}-{path_digest}.png"
    engines = [("pymupdf_pixmap", lambda: _render_pymupdf(pymupdf_page, target, max_render_pixels))]
    if password is None:
        engines.append(("poppler_pdftoppm", lambda: _render_poppler(pdf_path, page_index, output_dir, target, timeout, max_render_pixels, poppler_dir)))
    for engine, render in engines:
        try:
            if not render():
                warnings.append(f"visual_{engine}_unavailable")
                continue
        except OSError:
            raise
        except Exception as exc:
            warnings.append(_code(f"visual_{engine}_failed", exc))
            continue
        try:
            asset = _image_asset(target, output_dir, page_index, _bbox(pymupdf_page))
        except ValueError as exc:
            warnings.append(_code("visual_image_invalid", exc))
            continue
        if asset["width"] * asset["height"] > max_render_pixels:
            warnings.append("visual_image_exceeds_pixel_budget")
            continue
        asset["role"] = "fallback_full_page_visual"
        asset["recovery_engine"] = engine
        return asset
    warnings.append("full_page_visual_unavailable_original_pdf_retained_by_caller")
    return None


def _raw_result(pdf_path: Path, output_dir: Path, page_index: int, warnings: list[str]) -> dict[str, Any]:
    digest = _hash_file(pdf_path)
    target = output_dir / "assets" / f"original-{digest[:20]}.pdf"
    if not target.is_file() or _hash_file(target) != digest:
        with tempfile.NamedTemporaryFile(prefix="original-", suffix=".pdf", dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(pdf_path, temporary_path)
            os.replace(temporary_path, target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    asset = {
        "id": "original-pdf", "path": target.relative_to(output_dir).as_posix(),
        "media_type": "application/pdf", "content_type": "application/pdf",
        "sha256": digest, "size_bytes": target.stat().st_size,
        "role": "complete_original_document",
    }
    text = "[Original PDF preserved. This page could not be parsed or rendered; machine text is unavailable.]"
    return {
        "mode": "raw_pdf", "text": text,
        "blocks": [{"type": "raw_document", "text": text, "bbox": None, "line_ids": [], "asset_id": asset["id"], "recovered": True}],
        "assets": [asset], "warnings": warnings, "degraded": True,
        "capture_status": "original_preserved_page_unreadable", "page_index": page_index,
    }


def recover_page(
    pdf_path: Path, page_index: int, *, pymupdf_page: Any = None,
    password: str | None = None, reason: Any = None, output_dir: Path,
    max_render_pixels: int = DEFAULT_RENDER_PIXELS,
    timeout: float = DEFAULT_TOOL_TIMEOUT, poppler_dir: Path | None = None,
    include_visual: bool = False,
) -> dict[str, Any]:
    """Recover one zero-based page, writing any assets under ``output_dir``.

    Text fallback makes no structural fidelity claim.  Image/raw modes carry
    explicit unavailability messages, never fabricated source sentences.  If a
    password is supplied, external processes are skipped so a password is not
    exposed through their command lines.  The caller can pass an already
    authenticated PyMuPDF page for rendering in that case.
    """
    pdf_path = filesystem_path(pdf_path)
    output_dir = filesystem_path(output_dir)
    if page_index < 0:
        raise ValueError("page_index must be nonnegative")
    if not isinstance(max_render_pixels, int) or max_render_pixels < 1:
        raise ValueError("max_render_pixels must be a positive integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    warnings: list[str] = ["structured_extraction_unavailable"] if reason is not None else []

    def recovered_text(text: str, engine: str) -> dict[str, Any]:
        result = _text_result(text, engine, warnings, page_index)
        if include_visual:
            visual = preserve_page_visual(
                pdf_path, page_index, pymupdf_page=pymupdf_page,
                password=password, output_dir=output_dir,
                max_render_pixels=max_render_pixels, timeout=timeout,
                poppler_dir=poppler_dir, warnings=warnings,
            )
            if visual is not None:
                result["assets"].append(visual)
                result["blocks"].append({
                    "type": "page_image", "bbox": visual["bbox"],
                    "text": "[Full-page visual retained alongside recovered machine text.]",
                    "line_ids": [], "image_id": visual["id"],
                    "recovered": True, "synthetic": True,
                })
        return result

    if pymupdf_page is not None:
        try:
            text = decode_call(pymupdf_page.get_text, "text", sort=True)
            if _usable_text(text, "pymupdf", warnings):
                return recovered_text(text, "pymupdf_plain_text")
        except OSError:
            raise
        except Exception as exc:
            warnings.append(_code("pymupdf_text_failed", exc))
    try:
        text = decode_call(_pypdf_text, pdf_path, page_index, password)
        if _usable_text(text, "pypdf", warnings):
            return recovered_text(text, "pypdf")
    except OSError:
        raise
    except Exception as exc:
        warnings.append(_code("pypdf_text_failed", exc))
    if password is None:
        try:
            text = _poppler_text(pdf_path, page_index, output_dir, timeout, poppler_dir)
            if _usable_text(text, "poppler", warnings):
                return recovered_text(text, "poppler_pdftotext")
        except OSError:
            raise
        except Exception as exc:
            warnings.append(_code("poppler_text_failed", exc))
    else:
        warnings.append("external_password_process_skipped")
    path_digest = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:12]
    target = output_dir / "assets" / f"page-{page_index + 1:04d}-{path_digest}.png"
    rendered = False
    engine: str | None = None
    if password is None:
        try:
            rendered = _render_poppler(pdf_path, page_index, output_dir, target, timeout, max_render_pixels, poppler_dir)
            if rendered:
                engine = "poppler_pdftoppm"
            else:
                warnings.append("poppler_renderer_unavailable")
        except OSError:
            raise
        except Exception as exc:
            warnings.append(_code("poppler_render_failed", exc))
    if not rendered:
        try:
            rendered = _render_pymupdf(pymupdf_page, target, max_render_pixels)
            if rendered:
                engine = "pymupdf_pixmap"
            else:
                warnings.append("pymupdf_renderer_unavailable")
        except OSError:
            raise
        except Exception as exc:
            warnings.append(_code("pymupdf_render_failed", exc))
    if rendered:
        # Validate materialized bytes outside recovery handling so output
        # corruption / storage failures do not pretend to be successful.
        try:
            asset = _image_asset(target, output_dir, page_index, _bbox(pymupdf_page))
        except ValueError as exc:
            warnings.append(_code("rendered_image_invalid", exc))
            asset = None
        if asset is not None and asset["width"] * asset["height"] > max_render_pixels:
            warnings.append("rendered_image_exceeds_pixel_budget")
        elif asset is not None:
            text = f"[Page {page_index + 1}: page image preserved; machine-readable text is unavailable.]"
            return {
                "mode": "page_image", "text": text,
                "blocks": [{"type": "page_image", "text": text, "bbox": asset["bbox"], "line_ids": [], "image_id": asset["id"], "recovered": True}],
                "assets": [asset], "warnings": warnings, "degraded": True,
                "capture_status": "page_visual_preserved_no_machine_text",
                "recovery_engine": engine, "page_index": page_index,
            }
    return _raw_result(pdf_path, output_dir, page_index, warnings)


def recover_document(pdf_path: Path, *, output_dir: Path, reason: Any = None, **_: Any) -> dict[str, Any]:
    """Preserve a PDF whose page tree could not even be opened."""
    output_dir = filesystem_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    warnings = ["document_open_failed"] if reason is not None else ["document_unreadable"]
    result = _raw_result(filesystem_path(pdf_path), output_dir, 0, warnings)
    result["page_index"] = None
    result["capture_status"] = "original_preserved_document_unreadable"
    return result
