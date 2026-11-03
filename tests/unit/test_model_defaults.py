import inspect
import json
import os
import subprocess
import sys
import tomllib

import pytest

from app import __version__
from app.cli import _build_parser
from app.config import BASE_DIR, ModelDefaults, model_defaults
from app.llm_protocol import default_api_protocol
from app.main import create_run, rerun_reference_from_frozen_t5
from app.models import ModelRuntimeConfig


from pathlib import Path
from app.config import Settings


def test_environment_defaults_reach_models_api_and_cli():
    environment = dict(os.environ)
    environment.update({
        "PREPROCESS_API_BASE_URL": "https://example.invalid/v1/responses",
        "PREPROCESS_API_MODEL": "configured-model",
        "PREPROCESS_API_PROTOCOL": "responses",
        "PREPROCESS_REASONING_EFFORT": "medium",
        "PREPROCESS_API_KEY_ENV": "TEST_MODEL_CREDENTIAL",
        "TEST_MODEL_CREDENTIAL": "test-secret-never-return",
    })
    code = """
import inspect, json
from app.cli import _build_parser
from app.config import model_defaults
from app.main import create_run, public_model_defaults
from app.models import ModelRuntimeConfig
model = ModelRuntimeConfig()
args = _build_parser().parse_args(['run', '--input', 'manual.pdf', '--software-name', 'ExampleSoft', '--target-version', '1.0'])
assert args.api_base_url == model.base_url == inspect.signature(create_run).parameters['api_base_url'].default
assert args.api_model == model.model == inspect.signature(create_run).parameters['api_model'].default
assert args.reasoning_effort == model.reasoning_effort
assert args.api_key_env == 'TEST_MODEL_CREDENTIAL'
assert model.api_protocol == 'responses'
print(json.dumps(model_defaults.public_dict()))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=BASE_DIR, env=environment,
                            capture_output=True, text=True, check=True)
    public = json.loads(result.stdout)
    assert public["model"] == "configured-model"
    assert "test-secret-never-return" not in result.stdout
    assert "TEST_MODEL_CREDENTIAL" not in result.stdout


@pytest.mark.parametrize("url, protocol", [
    ("https://example.invalid/v1/responses/", "responses"),
    ("https://sogenport.com/v1", "chat_completions"),
    ("https://example.invalid/v1/chat/completions", "chat_completions"),
])
def test_protocol_inference_uses_endpoint_path(url, protocol):
    assert default_api_protocol(url) == protocol
    assert ModelRuntimeConfig(base_url=url).api_protocol == protocol


def test_explicit_saved_model_configuration_wins_over_process_defaults():
    saved = {
        "base_url": "https://saved.invalid/v1/responses",
        "model": "saved-model",
        "api_protocol": "chat_completions",
        "reasoning_effort": "low",
    }
    restored = ModelRuntimeConfig.model_validate(saved).model_dump()
    assert all(restored[key] == value for key, value in saved.items())
    assert saved["api_protocol"] == "chat_completions"


def test_removed_step_setting_is_absent_from_api_model_and_cli():
    parser = _build_parser()
    args = parser.parse_args(["run", "--input", "manual.pdf", "--software-name", "ExampleSoft", "--target-version", "1.0"])
    assert not hasattr(args, "max_steps")
    assert "max_steps" not in ModelRuntimeConfig.model_fields
    assert "max_steps" not in model_defaults.public_dict()
    for entrypoint in (create_run, rerun_reference_from_frozen_t5):
        assert "max_steps" not in inspect.signature(entrypoint).parameters
        assert inspect.signature(entrypoint).parameters["reasoning_effort"].default == model_defaults.reasoning_effort
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["run", "--input", "manual.pdf", "--max-steps", "12"])
    assert caught.value.code == 2


def test_old_saved_step_setting_is_ignored_without_breaking_run_loading():
    restored = ModelRuntimeConfig.model_validate({"max_steps": 12})
    assert "max_steps" not in restored.model_dump()
    assert "max_steps" not in restored.public_dict()


def test_legacy_task_trajectory_mode_is_no_longer_configurable(monkeypatch):
    monkeypatch.setenv("PREPROCESS_TASK_TRAJECTORY_MODE", "legacy")
    assert not hasattr(Settings(), "task_trajectory_mode")


def test_credential_selection_keeps_provider_isolation(monkeypatch):
    import app.main as main

    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setenv("NIMABO_API_KEY", "nimabo-test")
    monkeypatch.setenv("CUSTOM_MODEL_KEY", "custom-test")
    monkeypatch.setattr(main, "model_defaults", ModelDefaults(
        base_url="https://configured.invalid/v1", model="test-model", api_key_env="CUSTOM_MODEL_KEY"))
    assert main._resolve_api_key("", "https://configured.invalid/v1/") == "custom-test"
    assert main._resolve_api_key("", "https://api.nimabo.io/v1") == "nimabo-test"
    assert main._resolve_api_key("", "https://another.invalid/v1") == "openai-test"
    assert main._resolve_api_key(" submitted-test ", "https://api.nimabo.io/v1") == "submitted-test"


def test_project_version_comes_from_project_metadata():
    with (BASE_DIR / "pyproject.toml").open("rb") as stream:
        assert __version__ == tomllib.load(stream)["project"]["version"]


@pytest.mark.parametrize("endpoint", [
    "file:///local/config", "https://user:secret@example.invalid/v1",
])
def test_public_default_endpoint_rejects_non_http_urls_and_embedded_credentials(endpoint):
    with pytest.raises(ValueError, match="PREPROCESS_API_BASE_URL"):
        ModelDefaults(base_url=endpoint)


def test_cli_run_has_no_operation_path_option():
    parser = _build_parser()
    args = parser.parse_args(["run", "--input", "manual.pdf", "--software-name", "ExampleSoft", "--target-version", "1.0"])
    assert not hasattr(args, "operation_path")
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["run", "--input", "manual.pdf", "--operation-path", "gui"])
    assert caught.value.code == 2


def test_default_paths_are_isolated_inside_project():
    settings = Settings()

    assert BASE_DIR == Path(__file__).resolve().parents[2]
    assert settings.data_dir == BASE_DIR / "data"
    assert settings.output_dir == BASE_DIR / "output"
    assert not hasattr(settings, "agent_skill_dir")


def test_cli_has_one_output_root_for_all_exports(tmp_path):
    import argparse

    from app.cli import _build_parser, _runtime_settings

    root = tmp_path / "custom-output"
    configured = _runtime_settings(argparse.Namespace(output_dir=str(root)))
    assert configured.output_dir == root.resolve()
    assert not hasattr(configured, "agent_skill_dir")
    run_parser = next(action for action in _build_parser()._actions if isinstance(action, argparse._SubParsersAction)).choices["run"]
    assert "--output-dir" in run_parser._option_string_actions
    assert "--agent-skill-dir" not in run_parser._option_string_actions
