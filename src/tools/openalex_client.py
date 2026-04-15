"""
OpenAlex API 客户端
文档: https://docs.openalex.org
完全免费，不需要 API key
"""

import requests
from typing import Optional
from src.agent.state import Paper

BASE_URL = "https://api.openalex.org"


def _inverted_index_to_text(inverted_index: Optional[dict]) -> Optional[str]:
    """将 OpenAlex 的 inverted index 格式转换为普通文本"""
    if not inverted_index:
        return None
    # inverted_index 是 {word: [(position, term_count), ...]} 格式
    words_with_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos_info in positions:
                if isinstance(pos_info, tuple):
                    words_with_positions.append((pos_info[0], word))
                else:
                    words_with_positions.append((pos_info, word))
    # 按位置排序并拼接
    words_with_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words_with_positions)


def _get_author_names(authorships: list) -> list[str]:
    """从 authorships 中提取作者名字列表"""
    names = []
    for auth in authorships:
        author = auth.get("author")
        if author and author.get("display_name"):
            names.append(author["display_name"])
    return names


class OpenAlexClient:
    """OpenAlex API 客户端"""

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
        搜索论文

        Args:
            query: 搜索关键词
            year_filter: 年份下限（如 2021 表示 2021 年及以后）
            limit: 返回数量
            offset: 偏移量
        """
        # 构建过滤条件
        filters = []

        # 添加年份过滤
        if year_filter:
            filters.append(f"publication_year:>{year_filter}")

        # 构建查询字符串
        filter_str = ",".join(filters) if filters else None
        search_query = query if not filter_str else f"{query},{filter_str}"

        params = {
            "filter": f"default.search:{search_query}" if search_query else None,
            "sort": "relevance_score:desc,cited_by_count:desc",
            "per-page": min(limit, 100),  # OpenAlex 最大 100
            "page": (offset // limit) + 1 if offset else 1,
            "select": "id,title,publication_year,authorships,abstract_inverted_index,cited_by_count,primary_topic,type",
        }

        # 移除 None 值
        params = {k: v for k, v in params.items() if v is not None}

        response = self.session.get(f"{BASE_URL}/works", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("results", []):
            # 生成 paper_id（使用 OpenAlex ID）
            openalex_id = item.get("id", "").replace("https://openalex.org/", "")

            # 提取期刊/会议信息
            primary_topic = item.get("primary_topic")
            venue = primary_topic.get("display_name") if primary_topic else None

            papers.append(Paper(
                paper_id=f"openalex:{openalex_id}",
                title=item.get("title", ""),
                year=item.get("publication_year", 0) or 0,
                abstract=_inverted_index_to_text(item.get("abstract_inverted_index")),
                authors=_get_author_names(item.get("authorships", [])),
                venue=venue,
                citation_count=item.get("cited_by_count", 0) or 0,
                url=item.get("id"),  # OpenAlex URL
            ))

        return papers

    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """获取论文详情"""
        # paper_id 格式: "openalex:W123456789"
        if not paper_id.startswith("openalex:"):
            return None

        openalex_id = paper_id.replace("openalex:", "")
        url = f"{BASE_URL}/works/{openalex_id}"

        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        item = response.json()

        return Paper(
            paper_id=f"openalex:{openalex_id}",
            title=item.get("title", ""),
            year=item.get("publication_year", 0) or 0,
            abstract=_inverted_index_to_text(item.get("abstract_inverted_index")),
            authors=_get_author_names(item.get("authorships", [])),
            venue=item.get("display_name"),  # 需要获取正确的 venue
            citation_count=item.get("cited_by_count", 0) or 0,
            url=item.get("id"),
        )


# 便捷函数
def search_papers(query: str, year_filter: int = 5, limit: int = 10) -> list[Paper]:
    """搜索论文（便捷函数）"""
    client = OpenAlexClient()
    return client.search(query=query, year_filter=year_filter, limit=limit)
