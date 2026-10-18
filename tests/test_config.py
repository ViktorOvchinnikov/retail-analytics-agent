import retail_agent.config as config
from retail_agent.llm import GeminiModel


def test_default_gemini_model_is_supported_name(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert config.Settings.from_env().gemini_model == "gemini-3.1-flash-lite"


def test_gemini_requests_have_a_bounded_timeout(settings) -> None:
    model = GeminiModel(settings).model

    assert model.timeout == 30
    assert model.max_retries == 1
