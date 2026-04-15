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
    """Returns a LangChainTracer instance for LangSmith tracing."""
    settings = get_settings()
    if not settings.langsmith_tracing:
        return None
    # LangChainTracer uses LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT env vars
    return LangChainTracer()