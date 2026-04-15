from pydantic import BaseModel, Field
from typing import Optional

class Paper(BaseModel):
    paper_id: str
    title: str
    year: int
    abstract: Optional[str] = None
    abstract_translation: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    venue: Optional[str] = None  # 期刊/会议名称
    url: Optional[str] = None
    citation_count: int = 0

class SearchState(BaseModel):
    user_input: str = ""
    parsed_query: dict = Field(default_factory=dict)
    year_filter: int = 5  # 默认近5年
    sort_by: str = "relevance"  # relevance, citation_count, year
    papers: list[Paper] = Field(default_factory=list)
    selected_papers: list[Paper] = Field(default_factory=list)
    export_format: Optional[str] = None
    trace_url: Optional[str] = None
    error: Optional[str] = None
    formatted_output: Optional[str] = None  # Markdown 格式化输出