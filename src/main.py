import sys
import re
from src.agent.state import SearchState, Paper
from src.agent.graph import app
from src.utils.export import export_to_bibtex, export_to_csv
from src.tools.llm_client import summarize_paper_chinese
from rich.console import Console
from rich.markdown import Markdown

console = Console()

# 全局状态
current_state = {
    "papers": [],
    "selected_indices": set(),
}


def search_papers(query: str) -> list[Paper]:
    """执行论文搜索"""
    initial_state = SearchState(user_input=query)
    result = app.invoke(initial_state)
    return result.get("papers", [])


def format_current_results() -> str:
    """格式化当前搜索结果"""
    papers = current_state["papers"]
    selected = current_state["selected_indices"]

    if not papers:
        return "请先搜索论文。"

    lines = [f"## 搜索结果 (共 {len(papers)} 篇)\n"]
    lines.append("已选中的论文编号: " + (", ".join(str(i+1) for i in sorted(selected)) if selected else "无") + "\n")
    lines.append("---\n")

    for i, paper in enumerate(papers, 1):
        is_selected = i - 1 in selected
        marker = "✓ " if is_selected else "  "

        lines.append(f"### {marker}[{i}] {paper.title}")
        lines.append(f"**年份**: {paper.year} | **来源**: {paper.venue or 'N/A'} | **引用**: {paper.citation_count}")
        lines.append(f"**作者**: {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}")

        if paper.abstract:
            abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            lines.append(f"\n**摘要**: {abstract_preview}")

        if paper.url:
            lines.append(f"\n**链接**: {paper.url}")

        lines.append("\n---\n")

    lines.append("### 操作命令")
    lines.append("`select <编号>` - 选择论文，如 `select 1,3,5`")
    lines.append("`deselect <编号>` - 取消选择")
    lines.append("`export bibtex` - 导出选中论文为 BibTeX")
    lines.append("`export csv` - 导出选中论文为 CSV")
    lines.append("`summarize <编号>` - 生成论文摘要（中文）")
    lines.append("`new` - 新搜索")
    lines.append("`quit` - 退出")

    return "\n".join(lines)


def parse_select_command(cmd: str) -> list[int]:
    """解析 select 命令，返回论文编号列表（1-based）"""
    # 支持格式: select 1,3,5 或 select 1 3 5 或 select 1-3
    nums = re.findall(r'\d+', cmd)
    indices = []
    for n in nums:
        indices.append(int(n))
    return indices


def handle_command(user_input: str) -> bool:
    """
    处理用户命令
    返回 True 表示继续运行，False 表示退出
    """
    cmd = user_input.strip().lower()

    # 退出
    if cmd in ["quit", "exit", "q"]:
        console.print("再见!")
        return False

    # 新搜索
    if cmd == "new":
        return True  # 触发重新搜索

    # 选择论文
    if cmd.startswith("select "):
        indices = parse_select_command(cmd)
        for idx in indices:
            if 1 <= idx <= len(current_state["papers"]):
                current_state["selected_indices"].add(idx - 1)
        console.print(f"已选择: {', '.join(str(i+1) for i in sorted(current_state['selected_indices']))}")
        return True

    # 取消选择
    if cmd.startswith("deselect "):
        indices = parse_select_command(cmd)
        for idx in indices:
            if 1 <= idx <= len(current_state["papers"]):
                current_state["selected_indices"].discard(idx - 1)
        selected = current_state["selected_indices"]
        console.print(f"已选择: {', '.join(str(i+1) for i in sorted(selected)) if selected else '无'}")
        return True

    # 导出 BibTeX
    if cmd == "export bibtex":
        if not current_state["selected_indices"]:
            console.print("[yellow]请先选择论文[/yellow]")
            return True

        selected_papers = [current_state["papers"][i] for i in current_state["selected_indices"]]
        bibtex = export_to_bibtex(selected_papers)
        console.print("\n[bold]BibTeX 导出:[/bold]\n")
        console.print(bibtex)

        # 保存到文件
        with open("exported_papers.bib", "w") as f:
            f.write(bibtex)
        console.print("\n[green]已保存到 exported_papers.bib[/green]")
        return True

    # 导出 CSV
    if cmd == "export csv":
        if not current_state["selected_indices"]:
            console.print("[yellow]请先选择论文[/yellow]")
            return True

        selected_papers = [current_state["papers"][i] for i in current_state["selected_indices"]]
        csv = export_to_csv(selected_papers)
        console.print("\n[bold]CSV 导出:[/bold]\n")
        console.print(csv[:500] + "..." if len(csv) > 500 else csv)

        # 保存到文件
        with open("exported_papers.csv", "w") as f:
            f.write(csv)
        console.print("\n[green]已保存到 exported_papers.csv[/green]")
        return True

    # 生成摘要
    if cmd.startswith("summarize "):
        indices = parse_select_command(cmd)
        if not indices:
            console.print("[yellow]请指定论文编号，如 `summarize 1`[/yellow]")
            return True

        for idx in indices:
            if 1 <= idx <= len(current_state["papers"]):
                paper = current_state["papers"][idx - 1]
                console.print(f"\n[bold]正在生成论文 [{idx}] 的摘要...[/bold]")

                if paper.abstract:
                    try:
                        summary = summarize_paper_chinese(paper.title, paper.abstract)
                        console.print(f"\n[bold]《{paper.title}》摘要:[/bold]\n{summary}")
                    except Exception as e:
                        console.print(f"[red]生成摘要失败: {str(e)}[/red]")
                else:
                    console.print(f"[yellow]论文 [{idx}] 没有摘要，无法生成[/yellow]")

        return True

    # 未知命令
    console.print("[yellow]未知命令，输入 `new` 搜索论文，或 `quit` 退出[/yellow]")
    return True


def main():
    console.print("[bold green]Research Agent[/bold green] - 学术论文搜索智能体")
    console.print("输入 `quit` 退出\n")

    while True:
        try:
            user_input = console.input("[bold blue]你:[/bold blue] ").strip()

            if not user_input:
                continue

            # 先尝试作为命令处理
            if not handle_command(user_input):
                break

            # 如果不是命令，当作搜索查询处理
            console.print("\n[yellow]搜索中...[/yellow]\n")

            try:
                papers = search_papers(user_input)
                current_state["papers"] = papers
                current_state["selected_indices"] = set()

                if not papers:
                    console.print("[yellow]未找到相关论文[/yellow]")
                    continue

                # 显示结果
                output = format_current_results()
                md = Markdown(output)
                console.print(md)

            except Exception as e:
                console.print(f"[bold red]搜索出错:[/bold red] {str(e)}")

        except KeyboardInterrupt:
            console.print("\n\n再见!")
            break
        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
