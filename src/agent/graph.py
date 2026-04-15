from langgraph.graph import StateGraph, END
from src.agent.state import SearchState
from src.agent.nodes import parse_intent, execute_search
from src.tools.formatter import format_papers_markdown

def should_continue(state: SearchState) -> str:
    """决定是否继续"""
    if state.error:
        return "error"
    if not state.papers:
        return "no_results"
    return "format"

def format_node(state: SearchState) -> dict:
    """格式化节点"""
    return {"formatted_output": format_papers_markdown(state.papers)}

def build_graph():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("intent_parser", parse_intent)
    workflow.add_node("search_executor", execute_search)
    workflow.add_node("formatter", format_node)

    # 设置入口
    workflow.set_entry_point("intent_parser")

    # 添加边
    workflow.add_edge("intent_parser", "search_executor")
    workflow.add_edge("search_executor", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()

# 全局 app 实例
app = build_graph()