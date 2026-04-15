import re
from src.agent.state import SearchState

def parse_intent(state: SearchState) -> dict:
    """
    解析用户输入，提取搜索意图
    返回结构化查询参数
    """
    user_input = state.user_input
    parsed = {
        "query": user_input,
        "year_from": None,
        "sort_by": "relevance",
    }

    # 提取年份限制
    year_pattern = r"(\d{4})\s*年以后?|(\d{4})\s*年之后?|最近\s*(\d+)\s*年"
    match = re.search(year_pattern, user_input)
    if match:
        if match.group(1):
            parsed["year_from"] = int(match.group(1))
        elif match.group(3):
            from datetime import datetime
            current_year = datetime.now().year
            parsed["year_from"] = current_year - int(match.group(3))

    # 提取排序方式
    if "引用" in user_input or "citation" in user_input.lower():
        parsed["sort_by"] = "citation_count"
    elif "最新" in user_input or "新" in user_input:
        parsed["sort_by"] = "year"

    return parsed