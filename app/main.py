"""FastAPI 入口 —— 项目第二步：把 LLM 能力包装成 API 服务。

运行：uvicorn app.main:app --reload
然后打开 http://127.0.0.1:8000/docs 查看接口。
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.llm_client import chat
from app.rag import ask

app = FastAPI(title="文档知识库问答")


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"msg": "RAG 知识库问答服务运行中"}


@app.post("/chat")
def chat_api(req: ChatRequest):
    """最简对话接口。后续 RAG 会在这里接入检索逻辑。"""
    reply = chat(
        [
            {"role": "system", "content": "你是企业文档助手，回答要简洁准确"},
            {"role": "user", "content": req.question},
        ]
    )
    return {"answer": reply}


@app.post("/ask")
def ask_api(req: ChatRequest):
    """RAG 问答：检索知识库 → 拼接上下文 → 生成带出处的答案。"""
    return ask(req.question)
