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

def test_export_multiple_papers():
    papers = [
        Paper(paper_id="1", title="Paper 1", year=2023, abstract="A", authors=["A"], venue="CVPR", url="http://a.com"),
        Paper(paper_id="2", title="Paper 2", year=2022, abstract="B", authors=["B"], venue="ICCV", url="http://b.com"),
    ]
    bibtex = export_to_bibtex(papers)
    assert "Paper 1" in bibtex
    assert "Paper 2" in bibtex
    csv = export_to_csv(papers)
    assert "Paper 1" in csv
    assert "Paper 2" in csv