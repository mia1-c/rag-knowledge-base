# 企业知识库 RAG 问答系统

> LangChain + ChromaDB + LLM · 带**来源引用**的检索增强问答，支持 CLI 与 RESTful API 双模式

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="langchain" src="https://img.shields.io/badge/LangChain-%E2%89%A50.3-1C3C3C">
  <img alt="chromadb" src="https://img.shields.io/badge/ChromaDB-%E2%89%A50.5-FF6B6B">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-RESTful-009688?logo=fastapi&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

企业内部的制度文件、产品手册、FAQ 分散在各类文档中，关键词检索查不准、通用大模型又容易幻觉。
本项目实现一套基于**检索增强生成（RAG）**的知识库问答系统：支持 PDF / Markdown / TXT 多格式文档
自动入库、切片与向量化，用 ChromaDB 做语义检索，通过 Prompt 模板约束大模型**只依据知识库内容作答**，
并让每条答案附带**来源引用**，实现可追溯、可验证。

同时提供服务化能力：命令行交互模式用于快速验证，FastAPI RESTful 接口便于接入飞书、企业微信、
客服系统等上层应用。

---

## 目录

- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [调用 API](#调用-api)
- [核心设计](#核心设计)
- [场景迁移](#场景迁移)
- [项目结构](#项目结构)
- [注意事项](#注意事项)

---

## 功能特性

- 📂 **多格式文档**：PDF / Markdown / TXT（可扩展 Word、HTML）
- ✂️ **自动切片与向量化**：`RecursiveCharacterTextSplitter` 递归切分，保留上下文连贯性
- 🔍 **语义检索**：ChromaDB 向量相似度召回 + 可选的 MMR 多样性重排
- 💬 **LLM 生成**：Prompt 强制「只依据知识库回答」，显著降低幻觉
- 📎 **来源引用**：答案附带 `[来源: 文档名]`，可追溯可验证
- 🌐 **RESTful API**：FastAPI 提供 HTTP 接口，带 `/docs` 自动文档
- 🖥️ **命令行交互**：`python main.py` 即可开始对话
- 🔌 **模型可换**：OpenAI 兼容 API（DeepSeek / 通义 / Kimi…）或本地 Ollama 一键切换

---

## 技术架构

```
用户提问
    │
    ▼
┌───────────────┐      ┌──────────────────────┐      ┌────────────────┐
│  Query 处理    │ ───▶ │  向量检索（召回）      │ ───▶ │  LLM 生成回答   │
│  嵌入 / 改写    │      │  ChromaDB 相似度 Top-K │      │  + 来源引用     │
└───────────────┘      └──────────────────────┘      └────────────────┘
                                  ▲
                                  │
                       ┌──────────────────────┐
                       │  文档处理：切片 + 嵌入  │
                       └──────────┬───────────┘
                                  ▲
                       ┌──────────────────────┐
                       │  原始文档 PDF/MD/TXT  │
                       └──────────────────────┘
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置模型

复制 `.env.example` 为 `.env`，填入你的 LLM 配置：

```bash
# 方式一：OpenAI 兼容 API（推荐 DeepSeek，性价比高、支持 128k 上下文）
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 方式二：本地 Ollama（完全免费）
# LLM_API_KEY=ollama
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=qwen2.5:7b
```

> 💡 DeepSeek 不提供 embedding 接口，本项目嵌入默认走本地
> `BAAI/bge-small-zh-v1.5`（首次运行自动下载，约 100MB，已配置 hf-mirror 国内镜像加速）。
> 也可改用硅基流动 SiliconFlow 的免费 `bge-m3` API 嵌入。

可用切片 / 检索参数（`.env` 中可选）：

```bash
CHUNK_SIZE=500       # 切片长度
CHUNK_OVERLAP=50     # 切片重叠
TOP_K=4              # 召回片段数
COLLECTION_NAME=enterprise_kb
```

### 3. 导入文档

把你的企业文档（制度文件、产品手册、FAQ 等）放进 `data/` 目录，然后：

```bash
python scripts/ingest.py
```

### 4. 启动问答

```bash
# 命令行交互模式
python main.py

# API 服务模式
python main.py --api --port 8000
```

---

## 调用 API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "公司年假制度是怎么规定的？"}'
```

响应示例：

```json
{
  "question": "公司年假制度是怎么规定的？",
  "answer": "根据《员工手册》，入职满 1 年可享受 5 天年假…… [来源: 员工手册.pdf]",
  "sources": [{"source": "员工手册.pdf"}]
}
```

接口一览：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务状态 |
| GET | `/health` | 健康检查 |
| GET | `/docs` | 自动生成的接口文档 |
| POST | `/ask` | 问答接口 |

---

## 核心设计

### 文档切片
- 使用 `RecursiveCharacterTextSplitter` 递归切分，优先按段落 / 句子边界断开
- 默认 `chunk_size=500`、`chunk_overlap=50`，在检索粒度与上下文连贯性之间取平衡

### 向量检索
- 向量库：**ChromaDB**（本地持久化，零运维，适合中小规模知识库）
- 检索策略：相似度 Top-K（默认 K=4）+ 可选 MMR 多样性重排，避免召回内容高度重复

### 问答生成
- Prompt 模板注入召回片段，强制「**只依据知识库回答**，无依据时明确说明」
- 答案附带 `[来源: 文档名]` 引用标记，支持溯源复核
- LLM 与 Embedding 解耦配置，可分别指向不同服务

---

## 场景迁移

同一套架构可直接迁移到垂类问答场景，仅需替换 `data/` 语料与 Prompt：

| 场景 | 语料 | 关键改造点 |
|---|---|---|
| 企业制度 / 产品知识库 | 制度文件、产品手册 | 默认配置即可 |
| **高考志愿填报客服** | 各省投档线数据 | 增加**多约束解析**（省份+分数+地域+专业）+「冲-稳-保」分档逻辑，接入飞书机器人 |

> 「高考志愿填报客服」即是在本架构上完成的场景迁移：保留向量检索 + Prompt 约束 + 来源引用，
> 额外加入四维约束解析与分档推荐，并封装为飞书机器人对外提供自然语言问答。

---

## 项目结构

```
rag-knowledge-base/
├── main.py                # 入口：CLI 交互 / API 服务
├── config.py              # 配置管理（模型、向量库、切片与检索参数）
├── documents.py           # 文档加载与切片（PDF / MD / TXT）
├── retriever.py           # ChromaDB 向量库与检索器
├── chain.py               # RAG 链：Prompt 模板 + 生成 + 来源引用
├── scripts/
│   └── ingest.py          # 文档入库脚本（增量）
├── data/
│   └── sample.md          # 示例文档
├── kb_store/              # ChromaDB 持久化目录（自动生成，不入库）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 注意事项

- 首次运行需联网下载嵌入模型（也可切换为在线 embedding API）
- 文档更新后重新运行 `scripts/ingest.py` 即可增量入库
- `kb_store/` 与 `.env` 已加入 `.gitignore`，**请勿将 API Key 提交到仓库**
- 生产环境建议：向量库迁移至云端（Milvus / Qdrant），并补充权限与脱敏

---

## 项目描述（简历版）

> **基于 RAG 的企业知识库智能问答系统**（个人项目）
>
> 设计并实现了一套基于检索增强生成（RAG）的企业知识库问答系统，支持 PDF / Markdown / TXT
> 多格式文档自动入库、切片与向量化，采用 ChromaDB 实现语义检索，通过 Prompt 模板约束大模型
> 基于知识库内容生成带来源引用的回答，并提供 FastAPI RESTful 接口与命令行双交互模式。
> 系统可有效解决企业内部文档检索效率低、信息分散的问题，降低大模型幻觉风险，
> 支持快速扩展至垂类问答场景。
>
> **技术栈**：Python · LangChain · ChromaDB · FastAPI · 向量检索 · RAG · LLM

---

## License

MIT
