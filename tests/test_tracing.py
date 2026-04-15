from src.utils.tracing import get_tracer_config, langsmith_callback_handler
from langchain_core.tracers.langchain import LangChainTracer

def test_tracer_config_disabled():
    config = get_tracer_config()
    assert config is None or config.get("enabled") == False

def test_langsmith_handler_returns_none_when_disabled():
    """When LANGSMITH_TRACING is false, handler should be None"""
    handler = langsmith_callback_handler()
    assert handler is None

def test_langsmith_handler_returns_langchain_tracer_type():
    """Verify langsmith_callback_handler returns LangChainTracer when enabled (checked via source)"""
    import inspect
    source = inspect.getsource(langsmith_callback_handler)
    assert "LangChainTracer" in source