"""
OpenAlex API 客户端
文档: https://docs.openalex.org
完全免费，不需要 API key
"""

import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _parse_response(self, item: dict) -> Paper:
        """解析 API 响应为 Paper 对象"""
        openalex_id = item.get("id", "").replace("https://openalex.org/", "")
        primary_topic = item.get("primary_topic")
        venue = primary_topic.get("display_name") if primary_topic else None
        return Paper(
            paper_id=f"openalex:{openalex_id}",
            title=item.get("title", ""),
            year=item.get("publication_year", 0) or 0,
            abstract=_inverted_index_to_text(item.get("abstract_inverted_index")),
            authors=_get_author_names(item.get("authorships", [])),
            venue=venue,
            citation_count=item.get("cited_by_count", 0) or 0,
            url=item.get("id"),
        )

    def _search_single(
        self,
        query: str,
        year_filter: Optional[int],
        limit: int,
    ) -> list[Paper]:
        """执行单次搜索"""
        filter_parts = []
        if year_filter:
            filter_parts.append(f"publication_year:>{year_filter}")

        if filter_parts:
            filter_str = f"default.search:{query}," + ",".join(filter_parts)
        else:
            filter_str = f"default.search:{query}"

        params = {
            "filter": filter_str,
            "sort": "relevance_score:desc,cited_by_count:desc",
            "per-page": min(limit, 100),
            "select": "id,title,publication_year,authorships,abstract_inverted_index,cited_by_count,primary_topic,type",
        }

        response = self.session.get(f"{BASE_URL}/works", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        return [self._parse_response(item) for item in data.get("results", [])]

    def search(
        self,
        query: str,
        year_filter: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
        use_multi_search: bool = True,
    ) -> list[Paper]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            year_filter: 年份下限（如 2021 表示 2021 年及以后）
            limit: 返回数量
            offset: 偏移量
            use_multi_search: 是否使用多路搜索（对长查询分词并行搜索后合并）
        """
        # 对长查询使用多路搜索策略
        if use_multi_search and len(query.split()) >= 3:
            return self._multi_search(query, year_filter, limit, offset)

        # 短查询直接搜索
        results = self._search_single(query, year_filter, limit)
        return results[offset:offset + limit] if offset else results[:limit]

    def _generate_sub_queries(self, query: str) -> list[str]:
        """
        生成多个子查询，覆盖不同的关键词组合

        策略：
        1. 原始完整查询
        2. 每个核心词 + 相邻词组成二元组
        3. 连续 3-4 个词组成子查询
        """
        stop_words = {'and', 'or', 'in', 'on', 'under', 'with', 'for', 'the', 'a', 'an', 'of', 'to', 'from', 'by', 'using'}

        # 分词
        words = [w for w in query.lower().split() if w not in stop_words and len(w) > 2]

        sub_queries = []

        # 1. 原始完整查询
        sub_queries.append(query)

        # 2. 生成所有连续 3-4 词的子串
        for length in [3, 4]:
            for i in range(len(words) - length + 1):
                sub_q = ' '.join(words[i:i+length])
                if sub_q not in sub_queries:
                    sub_queries.append(sub_q)

        # 3. 特殊组合：确保包含 domain adaptation（因为这是关键概念）
        for w in words:
            if w not in ['domain', 'adaptation', 'point', 'cloud', 'object', 'detection']:
                combined = f"domain adaptation {w}"
                if combined not in sub_queries:
                    sub_queries.append(combined)

        # 去重并限制数量
        seen = set()
        unique_queries = []
        for q in sub_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:6]  # 最多 6 个子查询

    def _multi_search(
        self,
        query: str,
        year_filter: Optional[int],
        limit: int,
        offset: int,
    ) -> list[Paper]:
        """多路搜索：并行搜索多个子查询，合并结果"""
        sub_queries = self._generate_sub_queries(query)

        # 并行搜索，记录每篇论文被多少个子查询匹配到
        paper_scores = {}  # paper_id -> (paper, match_count)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_single, q, year_filter, limit): q
                for q in sub_queries
            }
            for future in as_completed(futures):
                try:
                    papers = future.result()
                    for paper in papers:
                        if paper.paper_id not in paper_scores:
                            paper_scores[paper.paper_id] = (paper, 0)
                        # 增加匹配计数
                        _, count = paper_scores[paper.paper_id]
                        paper_scores[paper.paper_id] = (paper, count + 1)
                except Exception:
                    pass

        # 转换为列表，按匹配分数排序（高的优先），再用引用数作为次级排序
        results = [
            paper for paper, _ in sorted(
                paper_scores.values(),
                key=lambda x: (x[1], x[0].citation_count if x[0].citation_count else 0),
                reverse=True
            )
        ]

        return results[offset:offset + limit] if offset else results[:limit]

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