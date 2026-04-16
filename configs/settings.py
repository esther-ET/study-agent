from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    semantic_scholar_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "research-agent"
    default_year_limit: int = 5
    default_page_size: int = 10
    minimax_api_key: str = ""
    deepseek_api_key: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
