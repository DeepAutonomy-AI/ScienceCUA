import json

import pytest

from app.runtime import AgentRuntimeConfigurationError
from app.runtime.dsh_client import native_web_environment, _payload_secrets
from app.runtime.dsh_events import RuntimeEventStream


@pytest.fixture(autouse=True)
def search_environment(monkeypatch):
    for name in ('DSH_WEB_SEARCH_PROVIDER', 'DEEPSEEK_API_KEY', 'PREPROCESS_SEARCH_BASE_URL',
                 'PREPROCESS_SEARCH_MODEL', 'PREPROCESS_SEARCH_API_KEY_ENV', 'PREPROCESS_SEARCH_PROTOCOL'):
        monkeypatch.delenv(name, raising=False)


def test_default_search_accepts_arbitrary_model_names_without_model_specific_rules():
    for model in ('writer-alpha', 'writer-beta'):
        environment = native_web_environment({'base_url': 'https://gateway.example/v1',
            'model': model, 'api_key': 'generation-secret'})
        config = json.loads(environment['RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG'])
        assert config == {'baseURL': 'https://gateway.example/v1', 'model': model,
                          'apiKey': 'generation-secret', 'protocol': 'auto'}


def test_independent_search_configuration_does_not_change_generation_model(monkeypatch):
    monkeypatch.setenv('PREPROCESS_SEARCH_BASE_URL', 'https://search.example/custom/v1/messages')
    monkeypatch.setenv('PREPROCESS_SEARCH_MODEL', 'retrieval-model')
    monkeypatch.setenv('PREPROCESS_SEARCH_API_KEY_ENV', 'TEST_SEARCH_KEY')
    monkeypatch.setenv('PREPROCESS_SEARCH_PROTOCOL', 'messages')
    monkeypatch.setenv('TEST_SEARCH_KEY', 'separate-search-secret')
    model = {'base_url': 'https://generation.example/v1', 'model': 'writer-model', 'api_key': 'generation-secret'}
    environment = native_web_environment(model)
    config = json.loads(environment['RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG'])
    assert config['model'] == 'retrieval-model' and config['apiKey'] == 'separate-search-secret'
    assert config['protocol'] == 'messages'
    assert model['model'] == 'writer-model' and model['api_key'] == 'generation-secret'
    secrets = _payload_secrets({'api_key': 'generation-secret', 'native_web_env': environment})
    assert 'generation-secret' in secrets and 'separate-search-secret' in secrets
    stream = RuntimeEventStream({'native_web_env': environment}, lambda _: None)
    assert stream.clean('separate-search-secret') == '[REDACTED]'


def test_credentials_are_not_implicitly_forwarded_to_another_origin(monkeypatch):
    monkeypatch.setenv('PREPROCESS_SEARCH_BASE_URL', 'https://another.example/v1')
    with pytest.raises(AgentRuntimeConfigurationError, match='not forwarded to another origin'):
        native_web_environment({'base_url': 'https://generation.example/v1', 'model': 'writer', 'api_key': 'secret'})


def test_missing_explicit_search_credential_does_not_fall_back_to_generation_key(monkeypatch):
    monkeypatch.setenv('PREPROCESS_SEARCH_API_KEY_ENV', 'TEST_ABSENT_SEARCH_KEY')
    monkeypatch.delenv('TEST_ABSENT_SEARCH_KEY', raising=False)
    with pytest.raises(AgentRuntimeConfigurationError, match='credential configuration'):
        native_web_environment({'base_url': 'https://gateway.example/v1', 'model': 'writer', 'api_key': 'secret'})


@pytest.mark.parametrize('base_url', ['https://user:secret@gateway.example/v1',
    'https://gateway.example/v1?key=private', 'file:///private'])
def test_search_endpoint_rejects_embedded_credentials_and_unsafe_urls(monkeypatch, base_url):
    monkeypatch.setenv('PREPROCESS_SEARCH_BASE_URL', base_url)
    with pytest.raises(AgentRuntimeConfigurationError, match='Search Base URL'):
        native_web_environment({'base_url': 'https://gateway.example/v1', 'model': 'writer', 'api_key': 'secret'})
