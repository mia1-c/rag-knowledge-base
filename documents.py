"""文档加载与切片模块：支持 PDF / Markdown / TXT / DOCX，自动按段落切分。"""
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# ========================
# 文档加载器映射
# ========================
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,  # Markdown 本质是纯文本，直接用 TextLoader 轻量高效
    ".markdown": TextLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


def load_single_file(file_path: Path) -> List[Document]:
    """加载单个文件，返回 Document 列表。"""
    ext = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        print(f"  [跳过] 不支持格式: {file_path.name}")
        return []

    try:
        # TextLoader / Markdown 需要指定编码
        if ext in (".txt", ".md", ".markdown"):
            loader = loader_cls(str(file_path), encoding="utf-8")
        else:
            loader = loader_cls(str(file_path))
        docs = loader.load()
        print(f"  [OK] {file_path.name} -> {len(docs)} 页/块")
        return docs
    except Exception as e:
        print(f"  [FAIL] {file_path.name}: {e}")
        return []


def load_documents(data_dir: Path = None) -> List[Document]:
    """加载 data_dir 下所有支持的文档。"""
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists():
        print(f"⚠️  目录不存在: {data_dir}")
        return []

    all_docs = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            fpath = Path(root) / fname
            all_docs.extend(load_single_file(fpath))

    print(f"\n[INFO] 共加载 {len(all_docs)} 个文档片段")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """将文档切片为小块，保留元数据。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] 切片完成，共 {len(chunks)} 个文本块")
    return chunks


if __name__ == "__main__":
    # 快速测试
    print("=== 文档加载测试 ===")
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        print(f"\n示例片段（前200字）:\n{chunks[0].page_content[:200]}")
