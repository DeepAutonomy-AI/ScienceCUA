"""Literal fidelity checks against the raw capture's first HTML5 parse."""
from __future__ import annotations

import pytest

from app.html_guided.pipeline import _Page, _projection
from app.html_guided.projection import preserve_initial_linefeeds, verify_projected_code
from app.html_sourcepart.extractor import parse_html


@pytest.mark.parametrize("boundary", ["body", "document"])
def test_body_projection_preserves_comments_and_escaped_literal_text(boundary):
    html = '<html><body>Literal &lt;widget&gt; and &amp; text.<!--<table><tr><td><img src="comment-only.png">Hidden draft</td></tr></table>--><h1>Real page</h1><p>Visible instructions.</p></body></html>'
    captured = parse_html(html)
    selected = captured.body if boundary == "body" else captured
    page = _Page("https://example.org/manual.html", html.encode(), "raw_fixture", 0)
    projected = _projection(page, html, selected).decode()
    restored = parse_html(projected)
    assert restored.find("widget") is None
    assert restored.find("table") is None
    assert restored.find("img") is None
    assert "Literal <widget> and & text." in restored.get_text()
    assert "Hidden draft" not in restored.get_text()
    assert '<!--<table>' in projected


@pytest.mark.parametrize("tag", ["pre", "listing", "textarea"])
@pytest.mark.parametrize("selected_region", ["body", "main"])
@pytest.mark.parametrize("normalize_dialog", [False, True])
def test_leading_blank_line_survives_projection_and_optional_second_parse(tag, selected_region, normalize_dialog):
    expected = "\n  run()\n    keep()"
    extra = '<dialog><img src=""></dialog>' if normalize_dialog else ""
    html = f'<html><head><title>Manual</title></head><body><main><h1>Manual</h1><{tag}>\n\n  run()\n    keep()</{tag}>' + extra + '</main></body></html>'
    captured = parse_html(html)
    original_serialization = str(captured)
    assert captured.find(tag).get_text() == expected
    region = captured.find(selected_region)
    adjustments, integrity = [], {}
    page = _Page("https://example.org/manual.html", html.encode("utf-8"), "raw_fixture", 0)
    projected = _projection(page, html, region, adjustments=adjustments, code_integrity=integrity).decode("utf-8")
    restored = parse_html(projected)
    assert restored.find(tag).get_text() == expected
    assert str(captured) == original_serialization
    assert integrity["status"] == "passed"
    assert integrity["code_nodes_checked"] == 1
    assert integrity["original_code_sha256"] == integrity["projected_code_sha256"]
    kind = "textarea_initial_lf_padding" if tag == "textarea" else "html5_initial_linefeed_guard"
    assert any(item["kind"] == kind and item["transport_only"] for item in adjustments)
    if tag == "textarea":
        assert "<!---->" not in restored.find(tag).get_text()
    if normalize_dialog:
        assert any(item["kind"] == "empty_closed_dialog_image" for item in adjustments)


@pytest.mark.parametrize("tag", ["pre", "listing", "textarea"])
def test_linefeed_helper_keeps_selected_node_and_repeated_round_trips_literal(tag):
    expected = "\n  run()"
    original = parse_html(f'<{tag}>\n\n  run()</{tag}>').find(tag)
    original_markup = str(original)
    current = original
    for _ in range(3):
        copied, adjustments = preserve_initial_linefeeds(current)
        assert copied is not current
        current = parse_html(str(copied)).find(tag)
        assert current.get_text() == expected
        assert str(original) == original_markup


def test_whole_document_clone_does_not_prepend_an_empty_html_tree():
    original = parse_html('<html lang="en"><head><title>Manual</title></head><body id="manual"><h1>Manual</h1><textarea>\n\n  input()</textarea><pre>\n\n  run()</pre></body></html>')
    original_markup = str(original)
    current = original
    for _ in range(3):
        copied, _ = preserve_initial_linefeeds(current)
        assert len(copied.find_all("html")) == 1
        assert copied.body["id"] == "manual"
        current = parse_html(str(copied))
        assert current.textarea.get_text() == "\n  input()"
        assert current.pre.get_text() == "\n  run()"
        assert str(original) == original_markup


def test_code_verification_rejects_corrupt_indentation_without_echoing_code():
    raw = '<main><pre><code>UNIQUE_PRIVATE_DEFINITION\n  run()</code></pre></main><aside><pre>Unselected external code</pre></aside>'
    good = '<main data-guided-projection="true"><pre><code>UNIQUE_PRIVATE_DEFINITION\n  run()</code></pre></main>'
    report = verify_projected_code(raw, "main", good)
    assert report["status"] == "passed"
    assert report["code_nodes_checked"] == 2
    bad = good.replace("\n  run()", "\nrun()")
    with pytest.raises(ValueError) as error:
        verify_projected_code(raw, "main", bad)
    assert "UNIQUE_PRIVATE_DEFINITION" not in str(error.value)
    assert "run()" not in str(error.value)
    assert "changed captured code" in str(error.value)
