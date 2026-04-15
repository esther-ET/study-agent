from src.agent.nodes.intent_parser import parse_intent
from src.agent.state import SearchState

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

def test_parse_recent_years():
    state = SearchState(user_input="最近3年的深度学习论文")
    result = parse_intent(state)
    assert "year_from" in result
    assert result["year_from"] >= 2023  # Assuming current year is 2026

def test_parse_sort_by_citation():
    state = SearchState(user_input="引用最多的目标检测论文")
    result = parse_intent(state)
    assert result["sort_by"] == "citation_count"

def test_parse_sort_by_year():
    state = SearchState(user_input="最新的深度学习论文")
    result = parse_intent(state)
    assert result["sort_by"] == "year"