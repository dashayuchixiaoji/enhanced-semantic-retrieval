"""LLM 调用封装 —— 项目第一步：先跑通这个文件。

使用 OpenAI 兼容接口，可切换 DeepSeek / 通义千问 / OpenAI 等。
运行自测：python -m app.llm_client
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 里的配置（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
load_dotenv()

# 禁用代理环境变量，避免国内 API 被系统代理干扰导致 SSL 连接中断
# （若你的网络确实需要代理才能访问该 API，请删除下面这段）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)


def get_client() -> OpenAI:
    """从环境变量创建客户端。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("请先复制 .env.example 为 .env，并填入 LLM_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,  # 网络抖动时自动重试
        timeout=60.0,   # 单次请求超时（秒）
    )


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.7) -> str:
    """最基础的对话：传入 messages，返回模型回复文本。

    messages 形如：
        [
            {"role": "system", "content": "你是企业文档助手"},
            {"role": "user", "content": "介绍一下项目背景"},
        ]
    """
    client = get_client()
    model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # 快速自测：直接运行 python -m app.llm_client
    reply = chat(
        [
            {"role": "system", "content": "你是一个简洁的助手"},
            {"role": "user", "content": "用一句话介绍什么是 RAG"},
        ]
    )
    print("模型回复：", reply)
