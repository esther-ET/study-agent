import requests
from typing import Optional
from src.agent.state import Paper, SearchState
from configs.settings import get_settings

BASE_URL = "https://api.semanticscholar.org/graph/v1"

class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().semantic_scholar_api_key
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(
        self,
        query: str,
        year_filter: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Paper]:
        """搜索论文"""
        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": "paperId,title,authors,year,abstract,venue,citationCount,url",
        }
        if year_filter:
            # 搜索近N年的论文
            params["year"] = f">{year_filter}"

        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            papers.append(Paper(
                paper_id=item["paperId"],
                title=item.get("title", ""),
                year=item.get("year", 0),
                abstract=item.get("abstract"),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                venue=item.get("venue"),
                citation_count=item.get("citationCount", 0),
                url=item.get("url"),
            ))
        return papers

    def get_paper_details(self, paper_id: str) -> Paper:
        """获取论文详情"""
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,authors,year,abstract,venue,citationCount,url",
        }
        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        item = response.json()
        return Paper(
            paper_id=item["paperId"],
            title=item.get("title", ""),
            year=item.get("year", 0),
            abstract=item.get("abstract"),
            authors=[a.get("name", "") for a in item.get("authors", [])],
            venue=item.get("venue"),
            citation_count=item.get("citationCount", 0),
            url=item.get("url"),
        )

def search_papers(query: str, year_filter: int = 5, limit: int = 10) -> list[Paper]:
    """便捷搜索函数"""
    client = SemanticScholarClient()
    return client.search(query=query, year_filter=year_filter, limit=limit)