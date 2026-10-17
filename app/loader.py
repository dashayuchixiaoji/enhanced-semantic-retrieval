"""文档加载与切分 —— RAG 第一步：把文档读进来，切成小段（chunks）。

支持 txt / md / pdf / docx。
运行自测：python -m app.loader
"""
import os
import re


def load_text(path: str) -> str:
    """按扩展名加载文档，返回纯文本。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return _read_text(path)
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    raise ValueError(f"暂不支持的格式：{ext}")


def _read_text(path: str) -> str:
    """读取文本文件，自动尝试常见编码，兼容 UTF-8 / GBK(GB18030) 等中文文档。

    按 utf-8 → gb18030 → utf-16 顺序尝试，避免因编码不匹配而报错。
    """
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 兜底：实在解不了就忽略坏字符，尽量读出来
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


# 句子结束标点（中文标点 + 英文问号/感叹号/分号）。
# 注意：刻意不含英文句点 "."，避免误切数字/版本号（如 3.14、v1.2）。
_SENTENCE_END = re.compile(r"[。！？!?；;]")


def split_sentences(text: str) -> list[str]:
    """按句子结束标点，把文本切成句子列表。"""
    sentences = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_text(text: str, chunk_size: int = 500, overlap_sentences: int = 1) -> list[str]:
    """按句子切分后，合并相邻句子成 chunk。

    相比按固定字符硬切，这里保证不把句子拦腰截断；
    相邻 chunk 之间重叠 overlap_sentences 个句子，避免语义丢失。
    """
    sentences = split_sentences(text)
    chunks = []
    buffer: list[str] = []   # 当前累积的句子
    buffer_len = 0           # 当前累积字数

    for s in sentences:
        # 超长句（超过 chunk_size）单独成 chunk，避免 chunk 过大
        if len(s) >= chunk_size:
            if buffer:
                chunks.append("".join(buffer))
                buffer, buffer_len = [], 0
            chunks.append(s)
            continue

        buffer.append(s)
        buffer_len += len(s)
        if buffer_len >= chunk_size:
            chunks.append("".join(buffer))
            # 保留最后 overlap_sentences 句，作为下一个 chunk 的重叠开头
            buffer = buffer[-overlap_sentences:] if overlap_sentences else []
            buffer_len = sum(len(x) for x in buffer)

    if buffer:
        chunks.append("".join(buffer))
    return chunks


if __name__ == "__main__":
    # 自测：构造多句话文本，验证"按句切分 + 合并"不会截断句子
    sample = "第一句话用来测试切分。第二句话验证句子边界。第三句话继续累积长度。" * 20
    chunks = split_text(sample, chunk_size=50, overlap_sentences=1)
    print(f"切分出了 {len(chunks)} 段，前 3 段：")
    for i, c in enumerate(chunks[:3]):
        print(f"[第{i + 1}段] {c}")
