"""向量库封装 —— 存向量 + 按语义检索。

使用 Chroma（本地向量库，数据落在项目根目录 chroma_db，无需额外服务）。
"""
import os

import chromadb

from app.embedding import embed_texts, embed_query

# 数据库目录固定在项目根目录（绝对路径），
# 避免不同进程（脚本 / FastAPI 服务）因启动目录不同而读不到同一份数据
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)


class VectorStore:
    def __init__(self, collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, chunks: list[str], source: str) -> None:
        """把 chunks 向量化后入库，source 记录来源文件名（用于答案引用）。"""
        embeddings = embed_texts(chunks)
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "index": i} for i in range(len(chunks))]
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """检索与 query 语义最接近的 top_k 段，返回 [{text, source, index}]。"""
        query_embedding = embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        return [
            {
                "text": doc,
                "source": meta.get("source", "未知来源"),
                "index": meta.get("index", 0),
            }
            for doc, meta in zip(docs, metas)
        ]

    def reset(self) -> None:
        """清空集合（重复入库前先清空，避免 id 冲突）。"""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
