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