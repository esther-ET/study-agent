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
        sort_by=sort_by,
    )

    return {"papers": papers}