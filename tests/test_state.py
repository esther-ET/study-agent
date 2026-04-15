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