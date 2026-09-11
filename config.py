"""配置管理模块：集中管理模型、向量库、切片参数。"""
import os
from pathlib import Path

# HuggingFace 镜像（国内加速，必须在使用模型前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# ===== 文档与向量库路径 =====
DATA_DIR = BASE_DIR / "data"          # 原始文档目录
KB_STORE_DIR = BASE_DIR / "kb_store"  # ChromaDB 持久化目录

# ===== 模型配置（支持 OpenAI 兼容 API / 本地 Ollama）=====
# 读取 .env 文件
from dotenv import load_dotenv
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# 嵌入模型（与 LLM 独立，通常使用小模型做向量化）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ===== 切片参数 =====
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ===== 检索参数 =====
TOP_K = int(os.getenv("TOP_K", "4"))  # 召回片段数量
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_kb")  # ChromaDB 集合名

if __name__ == "__main__":
    print("=== RAG 配置 ===")
    print(f"数据目录: {DATA_DIR}")
    print(f"向量库:   {KB_STORE_DIR}")
    print(f"LLM:      {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"Embed:    {EMBEDDING_MODEL}")
    print(f"切片:     {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}")
    print(f"检索 TopK:{TOP_K}")
