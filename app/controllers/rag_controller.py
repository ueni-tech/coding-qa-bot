"""
RAGコントローラー
LaravelでいうControllerに相当
UI層とサービス層を繋ぐビジネスロジックの制御
"""

from typing import Any
import streamlit as st

from app.services import pdf_service, rag_service, embedding_service
from app.repositories import vectorstore_repository
from app.utils.text_utils import split_text, count_tokens
from config.prompts import SUCCESS_MESSAGES, ERROR_MESSAGES
from config.settings import validate_api_key


def process_pdf_upload(
    pdf_file, chunk_size: int, chunk_overlap: int, embeddings
) -> dict[str, Any]:
    """
    PDFアップロードを処理する

    Args:
        pdf_file: アップロードされたPDFファイル
        chunk_size: チャンクサイズ
        chunk_overlap: チャンク重複サイズ
        embeddings: 埋め込むモデル

    Returns:
        処理結果の辞書
    """
    try:
        is_valid, message = pdf_service.validate_pdf(pdf_file)
        if not is_valid:
            return {"success": False, "error": message}

        text = pdf_service.extract_text_from_pdf(pdf_file)
        chunks = split_text(text, chunk_size, chunk_overlap)

        old_vs = st.session_state.get("vectorstore")
        new_vs = vectorstore_repository.reset_vectorstore(old_vs, chunks, embeddings)

        st.session_state.vectorstore = new_vs
        token_count = count_tokens(text)

        return {
            "success": True,
            "vectorstore": new_vs,
            "stats": {
                "text_length": len(text),
                "chunk_count": len(chunks),
                "token_count": token_count,
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def initialize_system() -> dict[str, Any]:
    """
    システムを初期化する

    Returns:
        初期化結果
    """
    result: dict[str, Any] = {"success": True, "warnings": []}

    if not validate_api_key():
        return {
            "success": False,
            "error": ERROR_MESSAGES["no_api_key"],
            "instruction": ERROR_MESSAGES["api_key_instruction"],
        }

    try:
        embeddings = embedding_service.create_embeddings()
        llm = rag_service.create_llm()

        result["embeddings"] = embeddings
        result["llm"] = llm
    except Exception as e:
        return {"success": False, "error": f"モデル初期化エラー: {str(e)}"}

    existing_vs = vectorstore_repository.load_vectorstore(embeddings)
    if existing_vs:
        result["vectorestore"] = existing_vs
        result["vectorestore_loaded"] = True

    return result


def search_and_answer(
    question: str, vectorstore, llm, top_k: int, mode: str = "rag"
) -> dict[str, Any]:
    """
    質問に対して検索・回答を行う

    Args:
        question: 質問文
        vectorstore: ベクトルストア
        llm: LLMモデル
        top_k: 検索結果
        mode: "rag"または"search_only"

    Returns:
        処理結果
    """
    try:
        if mode == "search_only":
            docs = rag_service.search_documents(vectorstore, question, top_k)
            return {
                "success": True,
                "mode": "search_only",
                "documents": docs,
                "count": len(docs),
            }
        else:
            rag_chain, retriever = rag_service.create_rag_chain(vectorstore, llm, top_k)

            answer = rag_service.generate_answer(rag_chain, question)
            docs = retriever.invoke(question)

            return {
                "success": True,
                "mode": "rag",
                "answer": answer,
                "documents": docs,
                "count": len(docs),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def reset_system() -> bool:
    """
    システムをリセットする

    Returns:
        リセット成功の可否
    """
    try:
        if "vectorstore" in st.session_state:
            vs = st.session_state.vectorstore
            if hasattr(vs, "_collection"):
                uuid = vs._collection.id
                vs._client.reset()
                vectorstore_repository.cleanup_old_collections(uuid)
            del st.session_state.vectorstore

        for key in ["rag_chain", "retriever", "llm"]:
            if key in st.session_state:
                del st.session_state[key]

        vectorstore_repository.delete_vectorstore()

        return True

    except Exception:
        return False


def update_model_settings(model_name: str, temperature: float) -> dict[str, Any]:
    """
    モデル設定を更新する

    Args:
        model_name: モデル名
        temperature: 温度パラメータ

    Returns:
        更新結果
    """
    try:
        new_llm = rag_service.create_llm(model_name, temperature)
        st.session_state.llm = new_llm

        for key in ["rag_chain", "retriever"]:
            if key in st.session_state:
                del st.session_state[key]

        return {"success": True, "model": model_name, "tempareture": temperature}

    except Exception as e:
        return {"success": False, "error": str(e)}
