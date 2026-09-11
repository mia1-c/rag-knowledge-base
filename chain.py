"""RAG 问答链：检索 + 提示词 + LLM 生成，输出带来源引用的回答。"""
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from retriever import get_retriever, similarity_search

# ========================
# 提示词模板：约束模型只依据知识库回答
# ========================
PROMPT_TEMPLATE = """你是一个严谨的企业知识库问答助手。
请**只依据**下面提供的知识库内容回答用户问题。
如果知识库中没有相关信息，请明确回答"知识库中未找到相关内容"，不要编造。

要求：
1. 回答要准确、简洁、有条理
2. 在回答末尾标注信息来源，格式：[来源: 文档名]
3. 如果多个来源信息不一致，指出差异

===== 知识库内容 =====
{context}

===== 用户问题 =====
{question}
"""


def format_context(docs: List) -> str:
    """将检索到的文档片段格式化为上下文文本。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段{i} | 来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_chain():
    """构建 RAG 问答链（LCEL 声明式写法）。"""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.1,  # 低温度保证回答确定性
        max_tokens=1024,
    )

    # LCEL 链：检索 → 格式化上下文 → 提示词 → LLM → 解析
    chain = (
        {
            "context": lambda x: format_context(similarity_search(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> Dict:
    """对外问答入口：返回答案 + 来源片段。"""
    # 1. 检索
    docs = similarity_search(question)
    if not docs:
        return {
            "question": question,
            "answer": "知识库为空或未检索到相关内容，请先导入文档。",
            "sources": [],
        }

    # 2. 生成
    chain = build_chain()
    answer = chain.invoke({"question": question})

    # 3. 整理来源
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    print("=== RAG 问答测试 ===")
    q = input("输入问题: ")
    result = ask(q)
    print(f"\n🤖 回答:\n{result['answer']}")
    print(f"\n📎 来源片段 {len(result['sources'])} 条:")
    for s in result["sources"]:
        print(f"  - {s['source']}")
