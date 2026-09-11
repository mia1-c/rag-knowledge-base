"""程序入口：支持命令行交互模式与 FastAPI API 服务模式。"""
import argparse
import sys
import io

# Windows 控制台 UTF-8 输出保护（仅在 CLI 模式使用，API 模式交给 uvicorn）
def _wrap_stdout():
    if sys.stdout and hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from chain import ask

# ========================
# 命令行交互模式
# ========================
def run_cli():
    _wrap_stdout()
    print("=" * 50)
    print("📚 企业知识库 RAG 问答系统")
    print("输入问题开始对话，输入 exit/quit 退出")
    print("=" * 50)
    while True:
        try:
            question = input("\n❓ 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        print("🤖 思考中...")
        result = ask(question)
        print(f"\n💡 回答:\n{result['answer']}")
        if result["sources"]:
            print("\n📎 参考来源:")
            for s in result["sources"]:
                print(f"   - {s['source']}")


# ========================
# API 服务模式
# ========================
def run_api(port: int):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(
        title="企业知识库 RAG 问答 API",
        description="基于 LangChain + ChromaDB 的检索增强生成问答服务",
        version="1.0.0",
    )

    class AskRequest(BaseModel):
        question: str

    class AskResponse(BaseModel):
        question: str
        answer: str
        sources: list

    @app.get("/")
    def root():
        return {"service": "RAG KB QA", "status": "ok", "docs": "/docs"}

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(req: AskRequest):
        """问答接口：POST {"question": "..."}"""
        return ask(req.question)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    print(f"🚀 API 服务启动: http://localhost:{port}")
    print(f"📖 接口文档: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="企业知识库 RAG 问答系统")
    parser.add_argument("--api", action="store_true", help="以 API 服务模式启动")
    parser.add_argument("--port", type=int, default=8000, help="API 端口（默认 8000）")
    args = parser.parse_args()

    if args.api:
        run_api(args.port)
    else:
        run_cli()
