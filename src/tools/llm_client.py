"""
LLM API 客户端 - 统一接口支持 MiniMax / DeepSeek

切换模型方法:
1. 修改 .env 文件中的 LLM_PROVIDER=minimax 或 LLM_PROVIDER=deepseek
2. 或修改 configs/settings.py 中的 llm_provider 默认值

支持的模型:
- MiniMax: minimax-m2.7, minimax-text-01
- DeepSeek: deepseek-chat, deepseek-coder
"""

import requests
from typing import Optional
from configs.settings import get_settings


# ==================== 配置 ====================

# 模型映射表，方便切换
MODEL_MAP = {
    "minimax": "minimax-m2.7",
    "deepseek": "deepseek-chat",
}

API_ENDPOINTS = {
    "minimax": "https://api.minimax.chat/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}


# ==================== 核心 API 调用 ====================

def get_current_provider() -> str:
    """获取当前配置的 LLM provider"""
    settings = get_settings()
    return settings.llm_provider or "minimax"


def get_api_key(provider: Optional[str] = None) -> str:
    """获取指定 provider 的 API key"""
    settings = get_settings()
    p = provider or get_current_provider()
    if p == "minimax":
        return settings.minimax_api_key
    return settings.deepseek_api_key


def get_model_name(provider: Optional[str] = None) -> str:
    """获取指定 provider 的模型名"""
    p = provider or get_current_provider()
    return MODEL_MAP.get(p, "minimax-m2.7")


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """
    统一的 LLM 调用接口

    Args:
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        provider: LLM 提供商，默认从 settings 读取
        model: 模型名，默认从 MODEL_MAP 读取
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        LLM 生成的文本
    """
    p = provider or get_current_provider()
    api_key = get_api_key(p)
    model_name = model or get_model_name(p)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        API_ENDPOINTS[p],
        headers=headers,
        json=data,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


# ==================== 便捷函数 ====================

def translate_to_chinese(text: str) -> str:
    """
    翻译文本为中文

    Args:
        text: 待翻译的英文文本

    Returns:
        中文翻译
    """
    system_prompt = "You are a professional translator. Translate the following text to Chinese, maintaining the original meaning and tone."
    return call_llm(
        prompt=f"Translate to Chinese:\n\n{text}",
        system_prompt=system_prompt,
    )


def translate_to_english(text: str) -> str:
    """
    翻译文本为英文

    Args:
        text: 待翻译的中文文本

    Returns:
        英文翻译
    """
    system_prompt = "You are a professional translator. Translate the following text to English, maintaining the original meaning and tone."
    return call_llm(
        prompt=f"Translate to English:\n\n{text}",
        system_prompt=system_prompt,
    )


def summarize_paper(paper_title: str, abstract: str) -> str:
    """
    生成论文摘要（英文）

    Args:
        paper_title: 论文标题
        abstract: 论文摘要

    Returns:
        英文摘要
    """
    system_prompt = "You are an academic paper summarizer. Given a paper's title and abstract, generate a concise summary (3-5 sentences) in English that captures the key contributions and findings."
    return call_llm(
        prompt=f"Title: {paper_title}\n\nAbstract: {abstract}\n\nPlease provide a concise summary of this paper:",
        system_prompt=system_prompt,
    )


def summarize_paper_chinese(paper_title: str, abstract: str) -> str:
    """
    生成论文摘要（中文）

    Args:
        paper_title: 论文标题
        abstract: 论文摘要

    Returns:
        中文摘要
    """
    system_prompt = "You are an academic paper summarizer. Given a paper's title and abstract, generate a concise summary (3-5 sentences) in Chinese that captures the key contributions and findings."
    return call_llm(
        prompt=f"Title: {paper_title}\n\nAbstract: {abstract}\n\n请用中文提供这篇论文的简洁摘要：",
        system_prompt=system_prompt,
    )


# ==================== 模型切换示例 ====================

if __name__ == "__main__":
    # 切换模型示例
    print("当前配置:")
    print(f"  Provider: {get_current_provider()}")
    print(f"  Model: {get_model_name()}")
    print()

    # 测试调用
    print("测试翻译:")
    result = translate_to_chinese("Deep learning is a subset of machine learning.")
    print(f"  {result}")
