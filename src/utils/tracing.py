import os
from langchain_core.tracers.langchain import LangChainTracer
from configs.settings import get_settings

def get_tracer_config() -> dict:
    settings = get_settings()
    if not settings.langsmith_tracing:
        return {"enabled": False}
    return {
        "enabled": True,
        "project_name": settings.langsmith_project,
        "api_key": settings.langsmith_api_key,
    }

def langsmith_callback_handler():
    """Returns a LangChainTracer instance for LangSmith tracing.

    When langsmith_tracing is enabled in settings, this function ensures
    LANGCHAIN_TRACING_V2 env var is set to enable LangChain's auto-tracing.
    """
    settings = get_settings()
    if not settings.langsmith_tracing:
        return None
    # Ensure LangChain's tracing env var is set so LangChainTracer actually traces
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or os.environ.get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return LangChainTracer()
