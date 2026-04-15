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