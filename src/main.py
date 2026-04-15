import sys
from src.agent.state import SearchState
from src.agent.graph import app
from src.utils.tracing import langsmith_callback_handler
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def main():
    console.print("[bold green]Research Agent[/bold green] - 学术论文搜索智能体")
    console.print("输入 `quit` 退出\n")

    while True:
        try:
            user_input = console.input("[bold blue]你:[/bold blue] ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("再见!")
                break

            if not user_input:
                continue

            # 构建初始状态
            initial_state = SearchState(user_input=user_input)

            # 执行 Agent
            with console.status("[bold yellow]搜索中..."):
                result = app.invoke(initial_state)

            # 输出结果
            console.print("\n")
            if result.get("formatted_output"):
                md = Markdown(result["formatted_output"])
                console.print(md)

            # 输出 trace URL
            if result.get("trace_url"):
                console.print(f"\n[dim]LangSmith Trace: {result['trace_url']}[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n再见!")
            break
        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
