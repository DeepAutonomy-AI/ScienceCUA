from fastapi.testclient import TestClient

from app.main import app


def test_health_and_web_surface_remain_available():
    with TestClient(app) as client:
        health = client.get("/api/health")
        page = client.get("/")
        script = client.get("/static/app.js")
        styles = client.get("/static/styles.css")
        defaults = client.get("/api/model-defaults")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert defaults.status_code == 200
    assert set(defaults.json()) == {"base_url", "model", "api_protocol", "reasoning_effort"}
    assert page.status_code == 200
    assert "ScienceCUA" in page.text
    assert 'id="websearch-toggle"' not in page.text
    assert 'operation_path_preference' not in page.text
    assert 'operation_path_preference' not in script.text
    assert 'data-evidence-view="tree"' in page.text
    assert 'data-evidence-view="flat"' in page.text
    assert 'id="evidence-tree-list"' in page.text
    assert script.status_code == 200
    assert "function renderEvidenceTrees" in script.text
    assert "function renderFlatEvidence" in script.text
    assert styles.status_code == 200
    assert ".evidence-tree-node" in styles.text


def test_websearch_request_is_rejected_explicitly():
    with TestClient(app) as client:
        response = client.post("/api/runs", data={"websearch": "true"})

    assert response.status_code == 422
    assert "Web Search is disabled" in response.json()["detail"]


def test_github_repository_url_is_outside_v2_scope():
    with TestClient(app) as client:
        response = client.post(
            "/api/runs",
            data={
                "software_name": "Example",
                "target_version": "1.0",
                "platform": "Linux",
                "ui_locale": "English",
                "urls_json": '[{"url":"https://github.com/example/repository"}]',
                "api_key": "test-only",
                "api_model": "test-model",
            },
        )

    assert response.status_code == 422
    assert "does not include GitHub repositories" in response.json()["detail"]
