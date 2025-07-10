"""
RAGサービス
RAGに関するビジネスロジックをここに集約
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

from config.settings import OPENAI_API_KEY, MODELS
from config.prompts import RAG_PROMPT_TEMPLATE
from app.utils.text_utils import format_docs
from app.services import embedding_service


def create_llm(model_name: str | None = None, teperature: float = 0.0) -> ChatOpenAI:
    """
    LLMモデルを作成する

    Args:
        model_name: モデル名（Noneの場合はデフォルト）
        temperature: 生成の多様性（0.0〜2.0）

    Returns:
        LLMモデル
    """
    if model_name is None:
        model_name = MODELS["chat"]["default"]

    return ChatOpenAI(
        model=model_name,
        temperature=teperature,
        openai_api_key=OPENAI_API_KEY,
    )


def create_rag_chain(vectorstore: Chroma, llm: ChatOpenAI, top_k: int = 3) -> tuple:
    """
    RAGチェーンを作成する

    Args:
        vectorstor: ベクトルストア
        llm: LLMモデル
        top_k: 検索件数

    Returns:
        (rag_chain, retriever)のタプル
    """
    retriever = vectorstore.as_retriever(
        seatch_type="similarity", search_kwargs={"k": top_k}
    )

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def search_documents(vectorstore: Chroma, query: str, top_k: int = 3) -> list:
    """
    ドキュメントを検索する（RAG生成なし）

    Args:
        vectorstore: ベクトルストア
        query: 検索クエリ
        top_k: 検索件数

    Returns:
        検索結果のドキュメントリスト
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    return retriever.invoke(query)


def generate_answer(rag_chain, question: str) -> str:
    """
    質問に対する回答を生成する

    Args:
        rag_chain: RAGチェーン
        question: 質問

    Returns:
        生成された回答
    """
    return rag_chain.invoke(question)


def get_chat_model_info() -> dict:
    """
    チャットモデルの情報を取得する

    Returns:
        チャットモデル情報の辞書
    """
    return {
        "available_models": MODELS["chat"]["options"],
        "default_model": MODELS["chat"]["default"],
        "model_descriptions": {
            "gpt-4o-mini": "最もコストパフォーマンスに優れたモデル",
            "gpt-4.1-nano": "2025年最新、最速・最安モデル",
            "gpt-4.1-mini": "GPT-4oより83%安価で高性能",
            "gpt-4o": "バランスの取れた汎用モデル",
            "gpt-4-turbo": "高性能・高精度モデル",
        },
    }
