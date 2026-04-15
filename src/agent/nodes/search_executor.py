from src.agent.state import SearchState
from src.tools.unified_search import UnifiedSearchClient

def execute_search(state: SearchState) -> dict:
    """
    执行论文搜索
    """
    query_data = state.parsed_query
    query = query_data.get("query", state.user_input)
    year_from = query_data.get("year_from")
    sort_by = query_data.get("sort_by", state.sort_by)

    client = UnifiedSearchClient()
    papers = client.search(
        query=query,
        year_filter=year_from or state.year_filter,
        limit=10,
    )

    # 按排序方式排序
    if sort_by == "citation_count":
        papers.sort(key=lambda p: p.citation_count if p.citation_count else 0, reverse=True)
    elif sort_by == "year":
        papers.sort(key=lambda p: p.year, reverse=True)

    return {"papers": papers}