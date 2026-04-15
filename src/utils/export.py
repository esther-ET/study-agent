import csv
import io
from typing import List
from src.agent.state import Paper

def paper_to_bibtex(paper: Paper) -> str:
    """将单篇论文转为 BibTeX 格式"""
    # 生成 cite key
    first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
    cite_key = f"{first_author}{paper.year}"

    # 判断类型（会议还是期刊）
    entry_type = "inproceedings" if paper.venue and any(
        conf in (paper.venue or "") for conf in ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICML"]
    ) else "article"

    lines = [
        f"@{entry_type}{{{cite_key},",
        f"  title = {{{paper.title}}},",
        f"  author = {{{' and '.join(paper.authors)}}},",
        f"  year = {{{paper.year}}},",
    ]

    if paper.venue:
        lines.append(f"  booktitle = {{{paper.venue}}},")
    if paper.url:
        lines.append(f"  url = {{{paper.url}}},")

    lines.append("}")
    return "\n".join(lines)

def export_to_bibtex(papers: List[Paper]) -> str:
    """导出多篇论文为 BibTeX"""
    return "\n\n".join(paper_to_bibtex(p) for p in papers)

def export_to_csv(papers: List[Paper]) -> str:
    """导出多篇论文为 CSV"""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "paper_id", "title", "year", "authors", "venue",
        "abstract", "citation_count", "url"
    ])
    writer.writeheader()
    for paper in papers:
        writer.writerow({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "authors": "; ".join(paper.authors),
            "venue": paper.venue or "",
            "abstract": paper.abstract or "",
            "citation_count": paper.citation_count,
            "url": paper.url or "",
        })
    return output.getvalue()