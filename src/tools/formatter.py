from src.agent.state import Paper
from typing import Optional

def translate_abstract_to_chinese(abstract: Optional[str]) -> Optional[str]:
    """
    将英文摘要翻译为中文
    注意：这里需要调用翻译API（如有道、百度翻译）
    暂时返回 None，后续集成翻译服务
    """
    # TODO: 集成翻译服务
    return None

def format_papers_markdown(
    papers: list[Paper],
    selected_indices: Optional[list[int]] = None,
) -> str:
    """
    将论文列表格式化为 Markdown
    """
    if not papers:
        return "未找到相关论文。"

    lines = [f"## 搜索结果 (共找到 {len(papers)} 篇相关论文)\n"]

    for i, paper in enumerate(papers, 1):
        is_selected = selected_indices and i - 1 in selected_indices
        selected_marker = "**[已选中]** " if is_selected else ""

        lines.append(f"### {i}. {selected_marker}{paper.title}")
        lines.append(f"**年份**: {paper.year} | **来源**: {paper.venue or 'N/A'} | **引用数**: {paper.citation_count}")
        lines.append(f"**作者**: {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}")

        # 英文摘要
        if paper.abstract:
            lines.append(f"\n**摘要 (EN)**:\n{paper.abstract}")
            # 中文翻译（如果有）
            if paper.abstract_translation:
                lines.append(f"\n**摘要 (CN)**:\n{paper.abstract_translation}")
        else:
            lines.append("\n**摘要**: 暂无")

        if paper.url:
            lines.append(f"\n**链接**: {paper.url}")

        lines.append("\n---\n")

    lines.append("### 操作选项")
    lines.append("1. 输入论文编号选择论文 (如: `select 1,3,5`)")
    lines.append("2. 输入 `export bibtex` 或 `export csv` 导出选中论文")
    lines.append("3. 输入 `summarize <编号>` 生成摘要")
    lines.append("4. 输入 `quit` 退出")

    return "\n".join(lines)