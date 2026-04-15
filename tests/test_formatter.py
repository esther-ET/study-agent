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

def test_format_multiple_papers():
    papers = [
        Paper(paper_id="1", title="Paper 1", year=2023, abstract="Abstract 1", authors=["A"], venue="CVPR", citation_count=10, url="http://example.com/1"),
        Paper(paper_id="2", title="Paper 2", year=2022, abstract="Abstract 2", authors=["B"], venue="ICCV", citation_count=20, url="http://example.com/2"),
    ]
    result = format_papers_markdown(papers)
    assert "Paper 1" in result
    assert "Paper 2" in result
    assert "2023" in result
    assert "2022" in result

def test_format_empty_papers():
    result = format_papers_markdown([])
    assert "未找到相关论文" in result

def test_format_with_selection():
    paper = Paper(
        paper_id="test-1",
        title="Test Paper",
        year=2023,
        abstract="Abstract",
        authors=["Author"],
        venue="CVPR",
        citation_count=10,
        url="http://example.com",
    )
    result = format_papers_markdown([paper], selected_indices=[0])
    assert "**[已选中]**" in result