"""RAG 编排 —— 把"加载 → 切分 → 向量化入库 → 检索 → 生成"串起来。

用法：
  from app.rag import ingest, ask
  ingest("docs/项目文档.pdf")     # 先把文档入库（一次即可）
  ask("项目预算多少？")            # 提问，返回带出处的答案
"""
from app.loader import load_text, split_text
from app.vector_store import VectorStore
from app.llm_client import chat


def ingest(file_path: str) -> int:
    """加载文档 → 切分 → 向量化 → 入库。返回入库的 chunk 数量。"""
    store = VectorStore()
    text = load_text(file_path)
    chunks = split_text(text)
    store.add(chunks, source=file_path)
    return len(chunks)


def ask(question: str, top_k: int = 4) -> dict:
    """提问 → 检索 → 拼接 prompt → 生成。返回 {answer, sources}。"""
    store = VectorStore()
    hits = store.search(question, top_k=top_k)
    context = "\n\n".join(f"[资料{i + 1}]\n{h['text']}" for i, h in enumerate(hits))

    prompt = (
        "请根据下面的资料回答问题。如果资料中没有答案，请明确说"
        "\"资料中未找到相关信息\"，不要编造。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}"
    )
    answer = chat([{"role": "user", "content": prompt}])
    sources = [h["source"] for h in hits]
    return {"answer": answer, "sources": sources}
