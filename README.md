# 增强型检索 · 语义检索

一个可完整运行的最小示例：把文档切分、向量化入库，再用**语义检索**（而不是关键词匹配）找出与问题最相关的片段，交给大模型生成**带出处**的回答。

传统关键词检索只会匹配字面重合的词，换个说法就搜不到；语义检索把问题和文档都映射到同一向量空间，按"意思相近"排序，因此"预算多少"也能命中原文里的"项目总金额约为……"。这个项目把这条链路摊开写清楚，方便逐层改造。

## 核心链路

```
文档 (pdf/docx/txt/md)
   │  loader：按扩展名解析 → 按句子切分 → 合并成 chunk（带重叠）
   ▼
Embedding 模型 → 向量
   │  vector_store：写入本地 Chroma（chroma_db/），元数据记录来源文件
   ▼
用户提问 → 同一模型编码为查询向量
   │  余弦相似度检索 Top-K
   ▼
拼装 Prompt（含"资料中没有就说没有"的约束）→ 大模型生成
   ▼
答案 + 来源文件列表
```

## 项目结构

```
.
├── app/
│   ├── loader.py        # 文档加载（pdf/docx/txt/md）+ 句子级切分
│   ├── embedding.py     # 向量化封装（批量调用，自动分批）
│   ├── vector_store.py  # Chroma 持久化封装：add / search / reset
│   ├── rag.py           # 编排：ingest 入库、ask 检索并生成
│   ├── llm_client.py    # 大模型调用封装
│   └── main.py          # FastAPI 服务
├── requirements.txt
└── .env.example
```

## 快速开始

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # 填入你的 LLM 与 Embedding API Key
```

**1. 先验证模型链路是否通**

```bash
python -m app.llm_client
```

**2. 文档入库**

```python
from app.rag import ingest
ingest("docs/项目文档.pdf")       # 返回入库的 chunk 段数
```

**3. 提问**

```python
from app.rag import ask
print(ask("项目预算多少？"))       # {'answer': ..., 'sources': [...]}
```

**4. 起接口服务**

```bash
uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000/docs ，可用接口：

| 接口 | 说明 |
| --- | --- |
| `POST /chat` | 纯对话（不检索），用于对比效果 |
| `POST /ask` | 语义检索 + 生成，返回答案与出处 |

## 可调的参数

| 位置 | 参数 | 作用 |
| --- | --- | --- |
| `loader.split_text` | `chunk_size` | 单段最大字符数，太大噪声多、太小语义被切碎 |
| `loader.split_text` | `overlap_sentences` | 相邻段重叠句数，缓解答案被切断 |
| `vector_store.search` | `top_k` | 检索条数，太少漏信息、太多稀释重点 |

## 工程注意点

- **API Key 不入库**：`.env` 已在 `.gitignore` 中，只提交 `.env.example` 占位。
- **向量库不提交**：`chroma_db/` 是本地生成产物，换机器重新 `ingest` 即可。
- **国内网络**：默认指向 DeepSeek / 阿里云百炼的兼容 OpenAI 接口，客户端已关闭代理环境变量读取（`trust_env=False`）。
- **抑制幻觉**：Prompt 中明确要求"资料中未找到相关信息"时不得编造，答案同时返回 `sources` 便于人工核对。

## 后续可扩展方向

- 混合检索：语义向量 + BM25 关键词召回后融合排序（RRF）
- 重排序：检索召回后用 rerank 模型精排，提升 Top-1 命中率
- 增量更新：按文档指纹去重、支持删除与重建单篇
- 多文档隔离：用 collection 或 metadata 过滤按业务域检索
