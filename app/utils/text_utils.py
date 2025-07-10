"""
テキスト処理ユーティリティ
"""

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    テキストをチャンクに分割する

    Args:
        text: 分割対象のテキスト
        chunk_size: チャンクサイズ
        chunk_overlap: チャンク間の重複サイズ

    Returns:
        分割されたテキストのリスト
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    return splitter.split_text(text)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    テキストのトークン数をカウントする

    Args:
        text: カウント対象のテキスト
        model: 使用するモデル名

    Returns:
        トークン数
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def format_docs(docs) -> str:
    """
    ドキュメントリストを文字列に整形する

    Args:
        docs: LangChainのドキュメントリスト

    Returns:
        整形された文字列
    """
    return "\n\n".join(doc.page_content for doc in docs)


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    テキストを指定長で切り詰める（プレビュー用）

    Args:
        text: 切り詰め対象のテキスト
        max_length: 最大文字数

    Returns:
        切り詰められたテキスト
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
