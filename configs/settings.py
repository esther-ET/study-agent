from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    semantic_scholar_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "research-agent"
    default_year_limit: int = 5
    default_page_size: int = 10

    # LLM 配置
    llm_provider: str = "minimax"  # "minimax" 或 "deepseek"
    minimax_api_key: str = ""
    deepseek_api_key: str = ""

    # Backwards-compatible aliases
    @property
    def langsmith_tracing(self) -> bool:
        return self.langchain_tracing_v2

    @property
    def langsmith_api_key(self) -> str:
        return self.langchain_api_key

    @property
    def langsmith_project(self) -> str:
        return self.langchain_project

@lru_cache
def get_settings() -> Settings:
    return Settings()
