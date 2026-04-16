"""
LLM API 客户端（MiniMax / DeepSeek）
"""
import requests
from typing import Optional
from configs.settings import get_settings

BASE_URL = "https://api.minimax.chat/v1"

class LLMClient:
    def __init__(self, provider: str = "minimax"):
        settings = get_settings()
        if provider == "minimax":
            self.api_key = settings.minimax_api_key
            self.base_url = "https://api.minimax.chat/v1"
        else:
            self.api_key = settings.deepseek_api_key
            self.base_url = "https://api.deepseek.com/v1"

        self.model = "minimax-m2.7" if provider == "minimax" else "deepseek-chat"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        调用 LLM 生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            LLM 生成的文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def summarize_paper(paper_title: str, abstract: str) -> str:
    """
    生成论文摘要（英文）

    Args:
        paper_title: 论文标题
        abstract: 论文摘要

    Returns:
        生成的摘要（英文）
    """
    client = LLMClient(provider="minimax")

    system_prompt = """You are an academic paper summarizer. Given a paper's title and abstract, generate a concise summary (3-5 sentences) in English that captures the key contributions and findings of the paper."""

    prompt = f"""Title: {paper_title}
Abstract: {abstract}

Please provide a concise summary of this paper:"""

    return client.generate(prompt, system_prompt)


def summarize_paper_chinese(paper_title: str, abstract: str) -> str:
    """
    生成论文摘要（中文）

    Args:
        paper_title: 论文标题
        abstract: 论文摘要

    Returns:
        生成的中文摘要
    """
    client = LLMClient(provider="minimax")

    system_prompt = """You are an academic paper summarizer. Given a paper's title and abstract, generate a concise summary (3-5 sentences) in Chinese that captures the key contributions and findings of the paper."""

    prompt = f"""Title: {paper_title}
Abstract: {abstract}

请用中文提供这篇论文的简洁摘要："""

    return client.generate(prompt, system_prompt)
