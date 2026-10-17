"""Embedding 封装 —— 把文字转成"向量"（一串代表语义的数字）。

统一走 OpenAI 兼容接口，通过 .env 配置切换云/本地服务：
  - 云（当前，通义千问）：EMBEDDING_BASE_URL 指向阿里云百炼的 OpenAI 兼容端点
  - 本地（未来）：把 EMBEDDING_BASE_URL 指向本地服务（如 ollama / TEI），
    并改 EMBEDDING_MODEL 即可，代码不用动。
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 禁用代理环境变量，避免国内 API 被系统代理干扰导致 SSL 连接中断
# （若你的网络确实需要代理才能访问该 API，请删除下面这段）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)


def _get_embedding_client() -> OpenAI:
    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if not api_key:
        raise ValueError("请先在 .env 里配置 EMBEDDING_API_KEY（阿里云百炼的 key）")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,
        timeout=60.0,
    )


def embed_texts(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """批量把多段文字转成向量，自动分批，返回顺序与输入一致。

    大文件会切成上千段，云 embedding 接口单次请求有条数/长度上限，
    这里按 batch_size 分批调用，避免超限报错。如仍报超限，把 batch_size 调小。
    """
    client = _get_embedding_client()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in resp.data)
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """把单个问题转成向量。"""
    return embed_texts([text])[0]
