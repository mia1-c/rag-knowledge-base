"""向量库与检索模块：基于 ChromaDB 实现文档向量化存储与语义检索。"""
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config import (
    KB_STORE_DIR,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    TOP_K,
    COLLECTION_NAME,
)


def get_embeddings() -> Embeddings:
    """初始化嵌入模型。

    优先级：
    1. 若配置了 EMBEDDING_API_KEY，使用 OpenAI 兼容 API（如硅基流动 bge-m3）
    2. 否则回退到本地 sentence-transformers 模型（完全离线免费）
    """
    if EMBEDDING_API_KEY and EMBEDDING_API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
        print(f"[Embedding] 使用 API: {EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )
    else:
        print("[Embedding] 使用本地模型: BAAI/bge-small-zh-v1.5")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )


def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Chroma:
    """获取（或创建）ChromaDB 向量库实例。"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(KB_STORE_DIR),
    )


def add_documents(
    docs: List[Document],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 2,
) -> Chroma:
    """向向量库写入文档切片（分批写入，降低内存峰值）。"""
    vectorstore = get_vectorstore(collection_name)

    # 分批写入，避免一次性嵌入大量文本导致内存峰值
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        vectorstore.add_documents(batch, ids=ids)
        print(f"  [入库] {min(i + batch_size, total)}/{total}")

    print(f"[OK] 已入库 {total} 个文本块")
    return vectorstore


def similarity_search(
    query: str,
    k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> List[Document]:
    """语义相似度检索，返回最相关的 k 个片段。"""
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=k)


def get_retriever(
    k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
):
    """返回可直接用于 LCEL 链的检索器。"""
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    print("=== 检索测试 ===")
    q = input("输入问题: ")
    results = similarity_search(q)
    for i, r in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"来源: {r.metadata.get('source', 'unknown')}")
        print(r.page_content[:200])
