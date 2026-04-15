from src.tools.openalex_client import OpenAlexClient, search_papers
from src.tools.arxiv_client import ArxivClient
from src.tools.unified_search import UnifiedSearchClient


def test_openalex_client_init():
    client = OpenAlexClient()
    assert client is not None
    assert hasattr(client, 'search')


def test_arxiv_client_init():
    client = ArxivClient()
    assert client is not None
    assert hasattr(client, 'search')


def test_unified_client_init():
    client = UnifiedSearchClient()
    assert client is not None
    assert hasattr(client, 'search')


def test_openalex_search_returns_list():
    """测试 OpenAlex 搜索返回列表"""
    client = OpenAlexClient()
    papers = client.search("machine learning", year_filter=2020, limit=3)
    assert isinstance(papers, list)
    assert len(papers) <= 3
    if papers:
        paper = papers[0]
        assert hasattr(paper, 'paper_id')
        assert hasattr(paper, 'title')
        assert paper.paper_id.startswith('openalex:')


def test_arxiv_search_returns_list():
    """测试 arXiv 搜索返回列表"""
    client = ArxivClient()
    papers = client.search("deep learning", year_filter=2020, limit=3)
    assert isinstance(papers, list)
    assert len(papers) <= 3
    if papers:
        paper = papers[0]
        assert hasattr(paper, 'paper_id')
        assert hasattr(paper, 'title')
        assert paper.paper_id.startswith('arxiv:')


def test_unified_search_openalex_only():
    """测试只使用 OpenAlex"""
    client = UnifiedSearchClient()
    papers = client.search("neural network", year_filter=2020, limit=5, preferred_source="openalex")
    assert isinstance(papers, list)
    if papers:
        assert all(p.paper_id.startswith('openalex:') for p in papers)


def test_unified_search_arxiv_only():
    """测试只使用 arXiv"""
    client = UnifiedSearchClient()
    papers = client.search("neural network", year_filter=2020, limit=5, preferred_source="arxiv")
    assert isinstance(papers, list)
    if papers:
        assert all(p.paper_id.startswith('arxiv:') for p in papers)
