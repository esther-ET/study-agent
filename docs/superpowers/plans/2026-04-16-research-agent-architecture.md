# Research Agent 技术架构文档

> **项目**: Research Agent - 学术论文搜索智能体
> **日期**: 2026-04-16
> **技术栈**: Python 3.11+, LangGraph, LangChain, OpenAlex API, arXiv API, MiniMax/DeepSeek LLM

---

## 1. 项目概述

### 1.1 目标

构建一个基于 LangGraph 的学术论文搜索智能体，支持：
- 语义模糊匹配（中文/英文关键词）
- 多数据源搜索（OpenAlex + arXiv）
- 年份限制和排序
- Markdown 格式输出
- 论文选择、导出（BibTeX/CSV）
- LLM 生成摘要（中英翻译）

### 1.2 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI (main.py)                          │
│                    循环交互：搜索/选择/导出/摘要                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │intent_parser │───▶│ search_executor  │───▶│  formatter   │   │
│  └──────────────┘    └──────────────────┘    └──────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         数据层                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │  OpenAlex    │    │      ArXiv       │    │  LLM Client  │   │
│  │   Client     │    │      Client      │    │ (MiniMax/DS) │   │
│  └──────────────┘    └──────────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 文件结构

```
study_agent/
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI 入口，循环交互
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               # SearchState, Paper 模型
│   │   ├── graph.py               # LangGraph StateGraph 构建
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── intent_parser.py   # 意图解析节点
│   │       └── search_executor.py # 搜索执行节点
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── openalex_client.py     # OpenAlex API 客户端（含多路搜索）
│   │   ├── arxiv_client.py        # arXiv API 客户端
│   │   ├── unified_search.py      # 统一搜索客户端
│   │   ├── llm_client.py          # LLM 统一接口（MiniMax/DeepSeek）
│   │   ├── semantic_scholar.py    # Semantic Scholar API（备用）
│   │   └── formatter.py           # Markdown 格式化
│   └── utils/
│       ├── __init__.py
│       ├── export.py              # BibTeX/CSV 导出
│       └── tracing.py             # LangSmith 追踪配置
├── configs/
│   └── settings.py                # Pydantic Settings 配置
├── tests/                         # 单元测试
├── docs/                          # 文档
│   ├── superpowers/
│   │   ├── specs/                 # 设计文档
│   │   └── plans/                 # 实现计划
│   └── 技术文档/                   # 入门教程
├── requirements.txt
├── .env.example
├── .env                           # 包含 API Key（不提交）
└── README.md
```

---

## 3. 核心模块详解

### 3.1 状态管理 (state.py)

```python
# src/agent/state.py
from pydantic import BaseModel, Field
from typing import Optional

class Paper(BaseModel):
    """论文数据模型"""
    paper_id: str
    title: str
    year: int
    abstract: Optional[str] = None
    abstract_translation: Optional[str] = None  # LLM 翻译
    authors: list[str] = Field(default_factory=list)
    venue: Optional[str] = None  # 期刊/会议
    url: Optional[str] = None
    citation_count: int = 0

class SearchState(BaseModel):
    """智能体状态"""
    user_input: str = ""                    # 用户原始输入
    parsed_query: dict = Field(default_factory=dict)  # 解析后的查询
    year_filter: int = 5                    # 年份过滤
    sort_by: str = "relevance"              # 排序方式
    papers: list[Paper] = Field(default_factory=list)   # 搜索结果
    selected_papers: list[Paper] = Field(default_factory=list)  # 选中论文
    export_format: Optional[str] = None
    error: Optional[str] = None
    formatted_output: Optional[str] = None  # Markdown 输出
```

**设计要点**：
- 使用 Pydantic 进行类型验证
- State 是mutable的，节点返回更新的字段即可
- 状态在节点间自动传递

---

### 3.2 意图解析节点 (intent_parser.py)

```python
# src/agent/nodes/intent_parser.py
import re
from src.agent.state import SearchState

def parse_intent(state: SearchState) -> dict:
    """
    解析用户输入，提取搜索参数
    返回需要更新的 state 字段
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

    return {"parsed_query": parsed}
```

**支持的语言模式**：
| 输入 | 解析结果 |
|------|----------|
| `深度学习` | query="深度学习", year_from=None |
| `2020年以后的论文` | year_from=2020 |
| `最近3年` | year_from=当前年份-3 |
| `引用最多的` | sort_by="citation_count" |
| `最新的` | sort_by="year" |

---

### 3.3 搜索执行节点 (search_executor.py)

```python
# src/agent/nodes/search_executor.py
from src.agent.state import SearchState
from src.tools.unified_search import UnifiedSearchClient

def execute_search(state: SearchState) -> dict:
    """执行论文搜索"""
    query_data = state.parsed_query
    query = query_data.get("query", state.user_input)
    year_from = query_data.get("year_from")
    sort_by = query_data.get("sort_by", state.sort_by)

    client = UnifiedSearchClient()
    papers = client.search(
        query=query,
        year_filter=year_from or state.year_filter,
        limit=10,
        sort_by=sort_by,
    )

    return {"papers": papers}
```

---

### 3.4 统一搜索客户端 (unified_search.py)

```python
# src/tools/unified_search.py
from src.tools.openalex_client import OpenAlexClient
from src.tools.arxiv_client import ArxivClient

class UnifiedSearchClient:
    def __init__(self, openalex_weight: int = 3, arxiv_weight: int = 1):
        self.openalex = OpenAlexClient()
        self.arxiv = ArxivClient()
        self.openalex_weight = openalex_weight
        self.arxiv_weight = arxiv_weight

    def search(self, query, year_filter=None, limit=10,
               preferred_source="both", sort_by="relevance"):
        # 支持多数据源合并或单一来源
        if preferred_source == "openalex":
            return self.openalex.search(query, year_filter, limit)
        elif preferred_source == "arxiv":
            return self.arxiv.search(query, year_filter, limit)
        else:
            # 合并搜索，权重 3:1
            openalex_count = int(limit * 3 / 4)
            arxiv_count = limit - openalex_count
            # ... 并行搜索后合并
```

---

### 3.5 OpenAlex 客户端与多路搜索 (openalex_client.py)

**问题**：OpenAlex 对长查询（≥3个关键词）的相关性排序会退化。

**解决方案**：多路搜索策略

```python
# src/tools/openalex_client.py
class OpenAlexClient:
    def search(self, query, year_filter=None, limit=10,
               use_multi_search=True):
        # 长查询启用多路搜索
        if use_multi_search and len(query.split()) >= 3:
            return self._multi_search(query, year_filter, limit)
        return self._search_single(query, year_filter, limit)

    def _multi_search(self, query, year_filter, limit):
        """多路搜索：分词 -> 并行搜索 -> 按匹配次数评分"""
        # 1. 分词生成子查询
        words = self._tokenize(query)  # 去除停用词
        sub_queries = self._generate_sub_queries(words)

        # 2. 并行搜索
        paper_scores = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._search_single, q, year_filter, limit): q
                      for q in sub_queries}
            for future in as_completed(futures):
                papers = future.result()
                for paper in papers:
                    # 记录每篇论文被多少个子查询匹配到
                    if paper.paper_id not in paper_scores:
                        paper_scores[paper.paper_id] = (paper, 0)
                    paper_scores[paper.paper_id] = (
                        paper,
                        paper_scores[paper.paper_id][1] + 1
                    )

        # 3. 按匹配分数降序排列
        results = sorted(
            [p for p, _ in paper_scores.values()],
            key=lambda x: x[1],  # 匹配分数
            reverse=True
        )
        return results[:limit]
```

**子查询生成示例**：
```
原始: "point cloud object detection adverse weather domain adaptation"
分词: ['point', 'cloud', 'object', 'detection', 'adverse', 'weather', 'domain', 'adaptation']
子查询:
  1. point cloud object detection adverse weather domain adaptation
  2. point cloud object
  3. cloud object detection
  4. object detection adverse
  5. detection adverse weather
  6. adverse weather domain
```

---

### 3.6 LLM 客户端 (llm_client.py)

**统一接口，支持模型切换**：

```python
# src/tools/llm_client.py
from configs.settings import get_settings

MODEL_MAP = {
    "minimax": "minimax-m2.7",
    "deepseek": "deepseek-chat",
}

API_ENDPOINTS = {
    "minimax": "https://api.minimax.chat/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

def call_llm(prompt, system_prompt=None, provider=None, model=None,
             temperature=0.7, max_tokens=2000):
    """统一 LLM 调用接口"""
    p = provider or get_settings().llm_provider
    api_key = get_api_key(p)
    model_name = model or MODEL_MAP.get(p)

    response = requests.post(
        API_ENDPOINTS[p],
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    return response.json()["choices"][0]["message"]["content"]

# 便捷函数
def translate_to_chinese(text):
    return call_llm(f"Translate to Chinese:\n\n{text}",
                    system_prompt="You are a professional translator.")

def summarize_paper_chinese(title, abstract):
    return call_llm(
        f"Title: {title}\n\nAbstract: {abstract}\n\n请用中文提供这篇论文的简洁摘要：",
        system_prompt="You are an academic paper summarizer."
    )
```

**切换模型**：修改 `.env` 文件
```bash
LLM_PROVIDER=minimax   # 或 deepseek
```

---

### 3.7 LangGraph 构建 (graph.py)

```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END
from src.agent.state import SearchState
from src.agent.nodes import parse_intent, execute_search
from src.tools.formatter import format_papers_markdown

def format_node(state: SearchState) -> dict:
    return {"formatted_output": format_papers_markdown(state.papers)}

def build_graph():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("intent_parser", parse_intent)
    workflow.add_node("search_executor", execute_search)
    workflow.add_node("formatter", format_node)

    # 设置入口
    workflow.set_entry_point("intent_parser")

    # 添加边
    workflow.add_edge("intent_parser", "search_executor")
    workflow.add_edge("search_executor", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()

app = build_graph()
```

**执行流程**：
```
app.invoke(SearchState(user_input="点云检测"))

State Update 1: {"parsed_query": {...}}
     ↓
State Update 2: {"papers": [Paper1, Paper2, ...]}
     ↓
State Update 3: {"formatted_output": "### 1. ..."}
```

---

### 3.8 CLI 入口 (main.py)

```python
# src/main.py
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def main():
    console.print("[bold green]Research Agent[/bold green]")

    while True:
        user_input = console.input("[bold blue]你:[/bold blue] ").strip()

        # 处理命令
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.startswith("select "):
            handle_select(user_input)
            continue
        if user_input == "export bibtex":
            handle_export_bibtex()
            continue
        if user_input.startswith("summarize "):
            handle_summarize(user_input)
            continue

        # 搜索
        papers = search_papers(user_input)
        display_results(papers)
```

**支持的命令**：
| 命令 | 功能 |
|------|------|
| `select 1,3,5` | 选择论文 |
| `deselect 2` | 取消选择 |
| `export bibtex` | 导出为 BibTeX |
| `export csv` | 导出为 CSV |
| `summarize 1` | 生成中文摘要 |
| `new` | 新搜索 |
| `quit` | 退出 |

---

### 3.9 导出功能 (export.py)

```python
# src/utils/export.py
def paper_to_bibtex(paper: Paper) -> str:
    """论文转 BibTeX 格式"""
    first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
    cite_key = f"{first_author}{paper.year}"

    # 根据会议名称判断类型
    entry_type = "inproceedings" if paper.venue and any(
        conf in (paper.venue or "") for conf in ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICML"]
    ) else "article"

    return f"""@{entry_type}{{{cite_key},
  title = {{{paper.title}}},
  author = {{{' and '.join(paper.authors)}}},
  year = {{{paper.year}}},
}}"""

def export_to_bibtex(papers: list[Paper]) -> str:
    return "\n\n".join(paper_to_bibtex(p) for p in papers)

def export_to_csv(papers: list[Paper]) -> str:
    # 使用 csv.DictWriter 导出
```

---

## 4. 配置管理 (settings.py)

```python
# configs/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    semantic_scholar_api_key: str = ""
    minimax_api_key: str = ""
    deepseek_api_key: str = ""

    # LLM 配置
    llm_provider: str = "minimax"  # minimax 或 deepseek

    # LangSmith 追踪
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "research-agent"

    # 默认参数
    default_year_limit: int = 5
    default_page_size: int = 10

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**`.env` 文件示例**：
```bash
# LLM 配置
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here

# LangSmith（可选）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
```

---

## 5. 搜索效果优化

### 5.1 问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 搜索结果不相关 | OpenAlex 长查询相关性退化 | 多路搜索，按匹配分数排序 |
| 排序不符合预期 | 硬编码按引用数排序 | 支持 sort_by 参数 |
| filter 构建错误 | 年份过滤被当作文本搜索 | 修复 filter 参数格式 |

### 5.2 多路搜索评分机制

```
论文 A 被 3 个子查询匹配到 → 匹配分数 = 3
论文 B 被 1 个子查询匹配到 → 匹配分数 = 1

排序：按匹配分数降序，相同分数按引用数降序
```

---

## 6. 技术亮点

### 6.1 分层解耦

```
CLI (main.py)
    ↓
StateGraph (graph.py)      ← 工作流编排
    ↓
Nodes (parser, executor)   ← 业务逻辑
    ↓
Clients (OpenAlex, LLM)    ← 外部服务
```

### 6.2 配置驱动

通过 `.env` 文件切换 LLM provider，无需修改代码。

### 6.3 状态管理

使用 Pydantic 进行类型验证，State 在节点间自动传递。

### 6.4 并行搜索

使用 ThreadPoolExecutor 并行查询多个数据源和子查询。

---

## 7. 后续扩展方向

1. **RAG 增强**：用论文摘要训练 RAG，提供更精准的搜索
2. **多智能体协作**：分离搜索智能体和摘要智能体
3. **Web 界面**：FastAPI + React 前端
4. **增量搜索**：记住搜索历史，支持翻页

---

## 8. 验证清单

```bash
# 1. 搜索功能
python -m src.main
# 输入: point cloud object detection
# 验证: 返回相关论文列表

# 2. 选择和导出
# select 1,3
# export bibtex
# 验证: 生成 BibTeX 文件

# 3. 摘要生成
# summarize 2
# 验证: 调用 MiniMax 生成中文摘要

# 4. 模型切换
# 修改 .env: LLM_PROVIDER=deepseek
# summarize 2
# 验证: 使用 DeepSeek 生成摘要
```

---

*文档版本：2026-04-16*
*项目地址：https://github.com/esther-ET/study-agent*
