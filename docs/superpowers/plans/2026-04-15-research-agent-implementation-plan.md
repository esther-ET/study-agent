# Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 LangGraph + Semantic Scholar API 的学术论文搜索智能体，支持语义模糊匹配、Markdown格式输出、导出 BibTeX/CSV、生成摘要、LangSmith 全链路追踪。

**Architecture:** 使用 LangGraph StateGraph 实现 Agent 循环：用户输入 → 意图解析 → API搜索 → 格式化输出 → 可选导出/摘要。各节点通过 State 共享数据，LangSmith 记录完整推理链。

**Tech Stack:** Python 3.11+, LangGraph, LangSmith, Semantic Scholar API, Rich (CLI美化), Pydantic (状态验证)

---

## 文件结构

```
study_agent/
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py          # SearchState 定义
│   │   ├── nodes.py          # 所有节点实现
│   │   └── graph.py          # StateGraph 构建
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── semantic_scholar.py  # API 调用封装
│   │   └── formatter.py         # Markdown 格式化
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tracing.py       # LangSmith 配置
│   │   └── export.py        # BibTeX/CSV 导出
│   └── main.py               # CLI 入口
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   ├── test_nodes.py
│   └── test_formatter.py
├── configs/
│   └── settings.py           # 配置管理
├── requirements.txt
├── .env.example              # 环境变量示例
└── README.md
```

---

## Task 1: 项目骨架搭建

**Files:**
- Create: `study_agent/requirements.txt`
- Create: `study_agent/.env.example`
- Create: `study_agent/src/__init__.py`
- Create: `study_agent/src/agent/__init__.py`
- Create: `study_agent/src/tools/__init__.py`
- Create: `study_agent/src/utils/__init__.py`
- Create: `study_agent/tests/__init__.py`
- Create: `study_agent/configs/__init__.py`
- Create: `study_agent/configs/settings.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
langgraph>=0.2.0
langchain-core>=0.3.0
langsmith>=0.1.0
requests>=2.31.0
rich>=13.7.0
pydantic>=2.5.0
python-dotenv>=1.0.0
bibtexparser>=1.4.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: 创建 .env.example**

```bash
# Semantic Scholar API
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here

# LangSmith Tracing (可选)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=research-agent
```

- [ ] **Step 3: 创建 configs/settings.py**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    semantic_scholar_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "research-agent"
    default_year_limit: int = 5
    default_page_size: int = 10

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: 提交**

```bash
cd ~/code/study_agent && git init && git add -A && git commit -m "chore: scaffold project structure"
```

---

## Task 2: 状态定义 (SearchState)

**Files:**
- Create: `src/agent/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: 编写测试 tests/test_state.py**

```python
from src.agent.state import SearchState, Paper

def test_search_state_defaults():
    state = SearchState(user_input="test query")
    assert state.user_input == "test query"
    assert state.year_filter == 5
    assert state.sort_by == "relevance"
    assert state.papers == []
    assert state.selected_papers == []

def test_paper_model():
    paper = Paper(
        paper_id="test-id",
        title="Test Paper",
        year=2023,
        abstract="Test abstract",
        url="https://example.com"
    )
    assert paper.title == "Test Paper"
    assert paper.year == 2023
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_state.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现 src/agent/state.py**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_state.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/agent/state.py tests/test_state.py
git commit -m "feat: add SearchState and Paper models"
```

---

## Task 3: LangSmith Tracing 配置

**Files:**
- Create: `src/utils/tracing.py`
- Modify: `src/agent/graph.py` (添加 tracer)

- [ ] **Step 1: 编写测试 tests/test_tracing.py**

```python
from src.utils.tracing import get_tracer_config, langsmith_callback_handler

def test_tracer_config_disabled():
    config = get_tracer_config()
    assert config is None or config.get("enabled") == False

def test_langsmith_handler_type():
    handler = langsmith_callback_handler()
    assert handler is not None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现 src/utils/tracing.py**

```python
from langsmith import traceable
from langchain_core.tracers.langsmith import LangSmithTracer
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
    settings = get_settings()
    if not settings.langsmith_tracing:
        return None
    return LangSmithTracer(
        project_name=settings.langsmith_project,
        api_key=settings.langsmith_api_key,
    )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_tracing.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/utils/tracing.py tests/test_tracing.py
git commit -m "feat: add LangSmith tracing configuration"
```

---

## Task 4: Semantic Scholar API 封装

**Files:**
- Create: `src/tools/semantic_scholar.py`
- Test: `tests/test_semantic_scholar.py`

- [ ] **Step 1: 编写测试 tests/test_semantic_scholar.py**

```python
from src.tools.semantic_scholar import SemanticScholarClient, search_papers

def test_client_init_without_key():
    client = SemanticScholarClient()
    assert client.api_key == ""

def test_search_papers_returns_list():
    # 注意：这个测试会真正调用 API，需要网络
    result = search_papers("machine learning", year_filter=2020, limit=5)
    assert isinstance(result, list)
    if result:
        paper = result[0]
        assert hasattr(paper, 'paper_id')
        assert hasattr(paper, 'title')
        assert hasattr(paper, 'year')
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_semantic_scholar.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现 src/tools/semantic_scholar.py**

```python
import requests
from typing import Optional
from src.agent.state import Paper, SearchState
from configs.settings import get_settings

BASE_URL = "https://api.semanticscholar.org/graph/v1"

class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().semantic_scholar_api_key
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(
        self,
        query: str,
        year_filter: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Paper]:
        """搜索论文"""
        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": "paperId,title,authors,year,abstract,venue,citationCount,url",
        }
        if year_filter:
            # 搜索近N年的论文
            params["year"] = f">{year_filter}"

        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            papers.append(Paper(
                paper_id=item["paperId"],
                title=item.get("title", ""),
                year=item.get("year", 0),
                abstract=item.get("abstract"),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                venue=item.get("venue"),
                citation_count=item.get("citationCount", 0),
                url=item.get("url"),
            ))
        return papers

    def get_paper_details(self, paper_id: str) -> Paper:
        """获取论文详情"""
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,authors,year,abstract,venue,citationCount,url",
        }
        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        item = response.json()
        return Paper(
            paper_id=item["paperId"],
            title=item.get("title", ""),
            year=item.get("year", 0),
            abstract=item.get("abstract"),
            authors=[a.get("name", "") for a in item.get("authors", [])],
            venue=item.get("venue"),
            citation_count=item.get("citationCount", 0),
            url=item.get("url"),
        )

def search_papers(query: str, year_filter: int = 5, limit: int = 10) -> list[Paper]:
    """便捷搜索函数"""
    client = SemanticScholarClient()
    return client.search(query=query, year_filter=year_filter, limit=limit)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_semantic_scholar.py -v`
Expected: PASS (需要网络)

- [ ] **Step 5: 提交**

```bash
git add src/tools/semantic_scholar.py tests/test_semantic_scholar.py
git commit -m "feat: add Semantic Scholar API client"
```

---

## Task 5: 意图解析节点 (IntentParser)

**Files:**
- Create: `src/agent/nodes/intent_parser.py`
- Modify: `src/agent/nodes.py` (导出)
- Test: `tests/test_intent_parser.py`

- [ ] **Step 1: 编写测试 tests/test_intent_parser.py**

```python
from src.agent.nodes.intent_parser import parse_intent

def test_parse_simple_query():
    state = SearchState(user_input="深度学习目标检测")
    result = parse_intent(state)
    assert "query" in result
    assert result["query"] == "深度学习目标检测"

def test_parse_year_from_query():
    state = SearchState(user_input="找2020年以后的目标检测论文")
    result = parse_intent(state)
    assert "year_from" in result
    assert result["year_from"] == 2020
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_intent_parser.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现 src/agent/nodes/intent_parser.py**

```python
import re
from src.agent.state import SearchState

def parse_intent(state: SearchState) -> dict:
    """
    解析用户输入，提取搜索意图
    返回结构化查询参数
    """
    user_input = state.user_input
    parsed = {
        "query": user_input,
        "year_from": None,
        "sort_by": "relevance",
    }

    # 提取年份限制
    year_pattern = r"(\d{4})\s*年以后?|(\d{4})\s*年之后?|最近\s*(\d+)\s*年"
    match = re.search(year_pattern, user_input)
    if match:
        if match.group(1):
            parsed["year_from"] = int(match.group(1))
        elif match.group(3):
            from datetime import datetime
            current_year = datetime.now().year
            parsed["year_from"] = current_year - int(match.group(3))

    # 提取排序方式
    if "引用" in user_input or "citation" in user_input.lower():
        parsed["sort_by"] = "citation_count"
    elif "最新" in user_input or "新" in user_input:
        parsed["sort_by"] = "year"

    return parsed
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_intent_parser.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/agent/nodes/intent_parser.py tests/test_intent_parser.py
git commit -m "feat: add intent parser node"
```

---

## Task 6: 搜索执行节点 (SearchExecutor)

**Files:**
- Create: `src/agent/nodes/search_executor.py`
- Modify: `src/agent/nodes.py`
- Test: `tests/test_search_executor.py`

- [ ] **Step 1: 编写测试 tests/test_search_executor.py**

```python
from src.agent.nodes.search_executor import execute_search
from src.agent.state import SearchState

def test_execute_search_updates_state():
    state = SearchState(
        user_input="machine learning",
        parsed_query={"query": "machine learning", "year_from": 2020},
        year_filter=5,
    )
    result = execute_search(state)
    assert "papers" in result
    assert isinstance(result["papers"], list)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_search_executor.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 src/agent/nodes/search_executor.py**

```python
from src.agent.state import SearchState
from src.tools.semantic_scholar import SemanticScholarClient

def execute_search(state: SearchState) -> dict:
    """
    执行论文搜索
    """
    query_data = state.parsed_query
    query = query_data.get("query", state.user_input)
    year_from = query_data.get("year_from")
    sort_by = query_data.get("sort_by", state.sort_by)

    client = SemanticScholarClient()
    papers = client.search(
        query=query,
        year_filter=year_from or state.year_filter,
        limit=10,
    )

    # 按排序方式排序
    if sort_by == "citation_count":
        papers.sort(key=lambda p: p.citation_count, reverse=True)
    elif sort_by == "year":
        papers.sort(key=lambda p: p.year, reverse=True)

    return {"papers": papers}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_search_executor.py -v`
Expected: PASS (需要网络)

- [ ] **Step 5: 提交**

```bash
git add src/agent/nodes/search_executor.py tests/test_search_executor.py
git commit -m "feat: add search executor node"
```

---

## Task 7: Markdown 格式化节点

**Files:**
- Create: `src/tools/formatter.py`
- Test: `tests/test_formatter.py`

- [ ] **Step 1: 编写测试 tests/test_formatter.py**

```python
from src.tools.formatter import format_papers_markdown
from src.agent.state import Paper

def test_format_single_paper():
    paper = Paper(
        paper_id="test-1",
        title="Test Paper",
        year=2023,
        abstract="This is a test abstract.",
        authors=["John Doe", "Jane Smith"],
        venue="CVPR",
        citation_count=100,
        url="https://example.com/paper/test-1",
    )
    result = format_papers_markdown([paper])
    assert "Test Paper" in result
    assert "2023" in result
    assert "CVPR" in result
    assert "This is a test abstract" in result
    assert "example.com" in result
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_formatter.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 src/tools/formatter.py**

```python
from src.agent.state import Paper
from typing import Optional

def translate_abstract_to_chinese(abstract: Optional[str]) -> Optional[str]:
    """
    将英文摘要翻译为中文
    注意：这里需要调用翻译API（如有道、百度翻译）
    暂时返回 None，后续集成翻译服务
    """
    # TODO: 集成翻译服务
    return None

def format_papers_markdown(
    papers: list[Paper],
    selected_indices: Optional[list[int]] = None,
) -> str:
    """
    将论文列表格式化为 Markdown
    """
    if not papers:
        return "未找到相关论文。"

    lines = [f"## 搜索结果 (共找到 {len(papers)} 篇相关论文)\n"]

    for i, paper in enumerate(papers, 1):
        is_selected = selected_indices and i - 1 in selected_indices
        selected_marker = "**[已选中]** " if is_selected else ""

        lines.append(f"### {i}. {selected_marker}{paper.title}")
        lines.append(f"**年份**: {paper.year} | **来源**: {paper.venue or 'N/A'} | **引用数**: {paper.citation_count}")
        lines.append(f"**作者**: {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}")

        # 英文摘要
        if paper.abstract:
            lines.append(f"\n**摘要 (EN)**:\n{paper.abstract}")
            # 中文翻译（如果有）
            if paper.abstract_translation:
                lines.append(f"\n**摘要 (CN)**:\n{paper.abstract_translation}")
        else:
            lines.append("\n**摘要**: 暂无")

        if paper.url:
            lines.append(f"\n**链接**: {paper.url}")

        lines.append("\n---\n")

    lines.append("### 操作选项")
    lines.append("1. 输入论文编号选择论文 (如: `select 1,3,5`)")
    lines.append("2. 输入 `export bibtex` 或 `export csv` 导出选中论文")
    lines.append("3. 输入 `summarize <编号>` 生成摘要")
    lines.append("4. 输入 `quit` 退出")

    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_formatter.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/tools/formatter.py tests/test_formatter.py
git commit -m "feat: add Markdown formatter"
```

---

## Task 8: 导出功能 (BibTeX/CSV)

**Files:**
- Create: `src/utils/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: 编写测试 tests/test_export.py**

```python
from src.utils.export import export_to_bibtex, export_to_csv
from src.agent.state import Paper

def test_export_to_bibtex():
    paper = Paper(
        paper_id="test-1",
        title="Test Paper Title",
        year=2023,
        abstract="Test abstract",
        authors=["John Doe", "Jane Smith"],
        venue="CVPR",
        url="https://example.com",
    )
    bibtex = export_to_bibtex([paper])
    assert "@inproceedings" in bibtex or "@article" in bibtex
    assert "Test Paper Title" in bibtex
    assert "2023" in bibtex

def test_export_to_csv():
    paper = Paper(
        paper_id="test-1",
        title="Test Paper",
        year=2023,
        abstract="Test abstract",
        authors=["John Doe"],
        venue="CVPR",
        citation_count=10,
        url="https://example.com",
    )
    csv = export_to_csv([paper])
    assert "title" in csv.lower()
    assert "Test Paper" in csv
    assert "2023" in csv
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 src/utils/export.py**

```python
import csv
import io
from typing import list
from src.agent.state import Paper

def paper_to_bibtex(paper: Paper) -> str:
    """将单篇论文转为 BibTeX 格式"""
    # 生成 cite key
    first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
    cite_key = f"{first_author}{paper.year}"

    # 判断类型（会议还是期刊）
    entry_type = "inproceedings" if paper.venue and any(
        conf in (paper.venue or "") for conf in ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICML"]
    ) else "article"

    lines = [
        f"@{entry_type}{{{cite_key},",
        f"  title = {{{paper.title}}},",
        f"  author = {{{' and '.join(paper.authors)}}},",
        f"  year = {{{paper.year}}},",
    ]

    if paper.venue:
        lines.append(f"  booktitle = {{{paper.venue}}},")
    if paper.url:
        lines.append(f"  url = {{{paper.url}}},")

    lines.append("}")
    return "\n".join(lines)

def export_to_bibtex(papers: list[Paper]) -> str:
    """导出多篇论文为 BibTeX"""
    return "\n\n".join(paper_to_bibtex(p) for p in papers)

def export_to_csv(papers: list[Paper]) -> str:
    """导出多篇论文为 CSV"""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "paper_id", "title", "year", "authors", "venue",
        "abstract", "citation_count", "url"
    ])
    writer.writeheader()
    for paper in papers:
        writer.writerow({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "authors": "; ".join(paper.authors),
            "venue": paper.venue or "",
            "abstract": paper.abstract or "",
            "citation_count": paper.citation_count,
            "url": paper.url or "",
        })
    return output.getvalue()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/utils/export.py tests/test_export.py
git commit -m "feat: add BibTeX and CSV export"
```

---

## Task 9: StateGraph 构建

**Files:**
- Create: `src/agent/graph.py`
- Modify: `src/agent/nodes.py` (统一导出)
- Test: `tests/test_graph.py`

- [ ] **Step 1: 编写测试 tests/test_graph.py**

```python
from src.agent.graph import build_graph, app

def test_graph_exists():
    assert app is not None
    assert hasattr(app, 'invoke')

def test_graph_nodes():
    graph = build_graph()
    assert "intent_parser" in graph.nodes
    assert "search_executor" in graph.nodes
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 src/agent/nodes.py**

```python
from src.agent.nodes.intent_parser import parse_intent
from src.agent.nodes.search_executor import execute_search

__all__ = ["parse_intent", "execute_search"]
```

- [ ] **Step 4: 实现 src/agent/graph.py**

```python
from langgraph.graph import StateGraph, END
from src.agent.state import SearchState
from src.agent.nodes import parse_intent, execute_search
from src.tools.formatter import format_papers_markdown
from src.utils.tracing import langsmith_callback_handler

def should_continue(state: SearchState) -> str:
    """决定是否继续"""
    if state.error:
        return "error"
    if not state.papers:
        return "no_results"
    return "format"

def build_graph():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("intent_parser", parse_intent)
    workflow.add_node("search_executor", execute_search)
    workflow.add_node("formatter", lambda state: {"formatted_output": format_papers_markdown(state.papers)})

    # 设置入口
    workflow.set_entry_point("intent_parser")

    # 添加边
    workflow.add_edge("intent_parser", "search_executor")
    workflow.add_edge("search_executor", END)

    return workflow.compile(
        callbacks=[langsmith_callback_handler()] if langsmith_callback_handler() else None,
        debug=True,
    )

# 全局 app 实例
app = build_graph()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent/graph.py src/agent/nodes.py tests/test_graph.py
git commit -m "feat: add LangGraph StateGraph"
```

---

## Task 10: CLI 主入口

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: 实现 src/main.py**

```python
import sys
from src.agent.state import SearchState
from src.agent.graph import app
from src.utils.tracing import langsmith_callback_handler
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def main():
    console.print("[bold green]Research Agent[/bold green] - 学术论文搜索智能体")
    console.print("输入 `quit` 退出\n")

    while True:
        try:
            user_input = console.input("[bold blue]你:[/bold blue] ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("再见!")
                break

            if not user_input:
                continue

            # 构建初始状态
            initial_state = SearchState(user_input=user_input)

            # 执行 Agent
            with console.status("[bold yellow]搜索中..."):
                result = app.invoke(initial_state)

            # 输出结果
            console.print("\n")
            if result.get("formatted_output"):
                md = Markdown(result["formatted_output"])
                console.print(md)

            # 输出 trace URL
            if result.get("trace_url"):
                console.print(f"\n[dim]LangSmith Trace: {result['trace_url']}[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n再见!")
            break
        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试运行**

```bash
cd ~/code/study_agent && python -m src.main
# 输入: 找机器学习相关的论文
# 验证: 应该输出 Markdown 格式的搜索结果
```

- [ ] **Step 3: 提交**

```bash
git add src/main.py
git commit -m "feat: add CLI main entry point"
```

---

## Task 11: README 文档

**Files:**
- Create: `study_agent/README.md`

- [ ] **Step 1: 编写 README.md**

```markdown
# Research Agent

基于 LangGraph + Semantic Scholar API 的学术论文搜索智能体。

## 功能特性

- 语义模糊匹配：输入中文或英文关键词，自动解析搜索意图
- 年份限制：默认近5年，支持动态指定
- 排序规则：按时间和期刊分量排序
- Markdown 输出：标题、年份、摘要（中英对照）、链接
- 导出功能：支持 BibTeX 和 CSV 格式
- 全链路追踪：LangSmith 可视化 Agent 推理过程

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

### 运行

```bash
python -m src.main
```

### 使用示例

```
你: 找自动驾驶点云目标检测domain adaptation相关论文

搜索结果 (共找到 10 篇相关论文)

### 1. Multi-Scale Point Cloud Segmentation for Autonomous Driving
**年份**: 2023 | **来源**: CVPR | **引用数**: 150

---

### 操作选项
1. 输入论文编号选择论文
2. 输入 `export bibtex` 或 `export csv` 导出选中论文
3. 输入 `quit` 退出
```

## 技术架构

```
用户输入 → IntentParser → SearchExecutor → Formatter → Markdown 输出
                ↓
          LangSmith Tracing
```

## 项目结构

```
study_agent/
├── src/
│   ├── agent/          # LangGraph 状态机
│   ├── tools/          # API 和格式化工具
│   └── utils/          # 工具函数
├── tests/              # 测试
└── configs/            # 配置
```

## License

MIT
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## 验证清单

| 任务 | 验证方式 |
|------|----------|
| 1. 项目骨架 | `ls -la src/` 确认目录结构 |
| 2. SearchState | `pytest tests/test_state.py -v` PASS |
| 3. LangSmith | 设置 `LANGSMITH_TRACING=true` 查看 trace |
| 4. API 封装 | `python -c "from src.tools.semantic_scholar import search_papers; print(len(search_papers('AI')))"` |
| 5. 意图解析 | `pytest tests/test_intent_parser.py -v` PASS |
| 6. 搜索执行 | `pytest tests/test_search_executor.py -v` PASS |
| 7. Markdown | `pytest tests/test_formatter.py -v` PASS |
| 8. 导出 | `pytest tests/test_export.py -v` PASS |
| 9. StateGraph | `pytest tests/test_graph.py -v` PASS |
| 10. CLI | `python -m src.main` 交互测试 |
| 11. README | 文件存在且内容完整 |

---

## 后续扩展

1. **翻译服务**：集成有道/百度翻译 API，翻译英文摘要为中文
2. **摘要生成**：调用 LLM 对论文生成简短总结
3. **Web 界面**：FastAPI 封装 + React 前端
4. **多数据源**：扩展 ArXiv、Google Scholar 等
