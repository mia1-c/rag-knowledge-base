#!/usr/bin/env python3
"""
精简版入库脚本 - 仅包含入库核心逻辑，减少内存占用
用法：python scripts/ingest.py
"""
import os, sys
from pathlib import Path

# 只在必要时才设置 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

COLLECTION_NAME = "enterprise_kb"
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "kb_store"


def load_documents() -> list[Document]:
    """加载 data 目录下所有支持的文档（PDF / MD / TXT）。"""
    docs = []
    for path in DATA_DIR.iterdir():
        if path.suffix.lower() == ".pdf":
            print(f"  [加载] {path.name} ...")
            try:
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"  [跳过] {path.name} 加载失败: {e}")
        elif path.suffix.lower() in (".md", ".txt"):
            print(f"  [加载] {path.name} ...")
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"  [跳过] {path.name} 加载失败: {e}")
    print(f"  [完成] 共加载 {len(docs)} 个文档")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """递归字符切片，默认每块 500 字，重叠 50 字。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  [切片] 共生成 {len(chunks)} 个文本块")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """初始化向量库（复用已有 collection，新文档追加）。"""
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=str(KB_DIR),
    )
    return vs


def main():
    print("=" * 50)
    print("RAG 知识库 - 文档入库")
    print("=" * 50)

    # 1. 加载
    docs = load_documents()
    if not docs:
        print("[错误] 未找到任何文档，退出。")
        return

    # 2. 切片
    chunks = split_documents(docs)

    # 3. 入库（分批，每批 2 个 chunk）
    vs = build_vectorstore(chunks)
    existing = vs._collection.count()
    print(f"  [存量] 当前向量库已有 {existing} 个文档块")

    BATCH_SIZE = 2
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        vs.add_documents(batch, ids=ids)
        print(f"  [入库] {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    total = vs._collection.count()
    print(f"\n✅ 入库完成！向量库共 {total} 个文档块")
    print(f"   (本次新增 {total - existing} 个)")


if __name__ == "__main__":
    main()
