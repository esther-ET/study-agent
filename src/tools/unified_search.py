"""
统一搜索客户端
结合 OpenAlex（综合）和 arXiv（CS/ML 专精）的搜索结果
"""

from typing import Optional
from src.agent.state import Paper
from src.tools.openalex_client import OpenAlexClient
from src.tools.arxiv_client import ArxivClient


class UnifiedSearchClient:
    """统一搜索客户端，同时查询多个数据源"""

    def __init__(self, openalex_weight: int = 3, arxiv_weight: int = 1):
        """
        Args:
            openalex_weight: OpenAlex 结果权重（返回比例）
            arxiv_weight: arXiv 结果权重
        """
        self.openalex = OpenAlexClient()
        self.arxiv = ArxivClient()
        self.openalex_weight = openalex_weight
        self.arxiv_weight = arxiv_weight

    def search(
        self,
        query: str,
        year_filter: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
        preferred_source: str = "openalex",
    ) -> list[Paper]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            year_filter: 年份下限
            limit: 返回数量
            offset: 偏移量
            preferred_source: 优先数据源 "openalex", "arxiv", 或 "both"
        """
        if preferred_source == "arxiv":
            return self.arxiv.search(query=query, year_filter=year_filter, limit=limit, offset=offset)
        elif preferred_source == "openalex":
            return self.openalex.search(query=query, year_filter=year_filter, limit=limit, offset=offset)
        else:
            # 合并两个数据源的结果
            total_weight = self.openalex_weight + self.arxiv_weight

            # 计算各数据源的数量
            openalex_count = int(limit * self.openalex_weight / total_weight)
            arxiv_count = limit - openalex_count

            # 并行查询（简化处理，串行查询）
            openalex_papers = self.openalex.search(
                query=query, year_filter=year_filter, limit=openalex_count, offset=offset
            )
            arxiv_papers = self.arxiv.search(
                query=query, year_filter=year_filter, limit=arxiv_count, offset=offset
            )

            # 合并并返回（OpenAlex 结果优先）
            combined = openalex_papers + arxiv_papers

            # 按引用数排序（如果有）
            combined.sort(key=lambda p: p.citation_count if p.citation_count else 0, reverse=True)

            return combined[:limit]

    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """获取论文详情"""
        if paper_id.startswith("openalex:"):
            return self.openalex.get_paper_details(paper_id)
        elif paper_id.startswith("arxiv:"):
            return self.arxiv.get_paper_details(paper_id)
        return None


# 便捷函数
def search_papers(
    query: str,
    year_filter: int = 5,
    limit: int = 10,
    preferred_source: str = "openalex",
) -> list[Paper]:
    """搜索论文（便捷函数）"""
    client = UnifiedSearchClient()
    return client.search(
        query=query,
        year_filter=year_filter,
        limit=limit,
        preferred_source=preferred_source,
    )
