"""
arXiv API 客户端
文档: https://arxiv.org/help/api
完全免费，不需要 API key
返回 Atom XML 格式
"""

import requests
from typing import Optional
from src.agent.state import Paper
from datetime import datetime
import xml.etree.ElementTree as ET

BASE_URL = "https://export.arxiv.org/api/query"


def _parse_arxiv_id(entry_id: str) -> str:
    """从 entry id URL 提取 arXiv ID"""
    # 格式: http://arxiv.org/abs/2306.04338v1
    return entry_id.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")


def _parse_authors(entry: ET.Element) -> list[str]:
    """从 entry 中提取作者名字"""
    authors = []
    for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name_elem = author.find("{http://www.w3.org/2005/Atom}name")
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text)
    return authors


def _parse_year(entry: ET.Element) -> int:
    """从 entry 中提取年份"""
    published = entry.find("{http://www.w3.org/2005/Atom}published")
    if published is not None and published.text:
        try:
            return datetime.fromisoformat(published.text[:10]).year
        except (ValueError, TypeError):
            pass
    return 0


class ArxivClient:
    """arXiv API 客户端（专注 CS/ML 论文）"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Research-Agent/1.0 (Python; mail@example.com)"
        })

    def search(
        self,
        query: str,
        year_filter: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Paper]:
        """
        搜索 arXiv 论文

        Args:
            query: 搜索关键词
            year_filter: 年份下限
            limit: 返回数量
            offset: 偏移量
        """
        # arXiv 搜索语法: https://arxiv.org/help/api/user-manual#query_details
        # all: 搜索所有字段
        search_parts = [f"all:{query}"]

        if year_filter:
            search_parts.append(f"all:{year_filter}")

        search_query = "+AND+".join(search_parts)

        params = {
            "search_query": f"all:{query}",  # 简化搜索
            "start": offset,
            "max_results": min(limit, 100),  # arXiv 最大 100
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        if year_filter:
            # arXiv 支持 date filters
            params["dateFilterBy"] = "submittedDate"
            params["dateFromDate"] = f"{year_filter}-01-01"

        response = self.session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        # 解析 Atom XML
        root = ET.fromstring(response.text)

        papers = []
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            entry_id = entry.find("{http://www.w3.org/2005/Atom}id")
            title = entry.find("{http://www.w3.org/2005/Atom}title")
            summary = entry.find("{http://www.w3.org/2005/Atom}summary")
            published = entry.find("{http://www.w3.org/2005/Atom}published")
            link = entry.find("{http://www.w3.org/2005/Atom}link[@rel='alternate']")

            # 提取 arXiv ID
            arxiv_id = _parse_arxiv_id(entry_id.text) if entry_id is not None else ""

            # 处理摘要（移除多余空白）
            abstract_text = ""
            if summary is not None and summary.text:
                abstract_text = " ".join(summary.text.split())

            papers.append(Paper(
                paper_id=f"arxiv:{arxiv_id}",
                title=title.text.strip() if title is not None and title.text else "",
                year=_parse_year(entry),
                abstract=abstract_text,
                authors=_parse_authors(entry),
                venue="arXiv",  # arXiv 是预印本平台
                citation_count=0,  # arXiv API 不提供引用数
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            ))

        return papers

    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """获取论文详情"""
        if not paper_id.startswith("arxiv:"):
            return None

        arxiv_id = paper_id.replace("arxiv:", "")
        # 通过 ID list 查询获取详细信息
        params = {
            "id_list": arxiv_id,
        }

        response = self.session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")

        if entry is None:
            return None

        entry_id = entry.find("{http://www.w3.org/2005/Atom}id")
        title = entry.find("{http://www.w3.org/2005/Atom}title")
        summary = entry.find("{http://www.w3.org/2005/Atom}summary")

        arxiv_id = _parse_arxiv_id(entry_id.text) if entry_id is not None else ""

        return Paper(
            paper_id=f"arxiv:{arxiv_id}",
            title=title.text.strip() if title is not None and title.text else "",
            year=_parse_year(entry),
            abstract=" ".join(summary.text.split()) if summary is not None and summary.text else "",
            authors=_parse_authors(entry),
            venue="arXiv",
            citation_count=0,
            url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        )


# 便捷函数
def search_papers(query: str, year_filter: int = 5, limit: int = 10) -> list[Paper]:
    """搜索论文（便捷函数）"""
    client = ArxivClient()
    return client.search(query=query, year_filter=year_filter, limit=limit)
