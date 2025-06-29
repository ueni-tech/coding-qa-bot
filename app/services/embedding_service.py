"""
埋め込みサービス
埋め込みベクトルに関する処理を管理
将来的な拡張（カスタム埋め込み、ベクトル演算など）に備えた設計
"""

import numpy as np
from typing import Any
from langchain_openai import OpenAIEmbeddings

from config.settings import OPENAI_API_KEY, MODELS


def create_embeddings(model_name: str | None = None, **kwargs) -> OpenAIEmbeddings:
    """
    埋め込みモデルを作成する

    Args:
        model_name: モデル名（Noneの場合はデフォルト使用）
        **kwargs: 追加のモデルパラメータ

    Returns:
        埋め込みモデルインスタンス
    """
    if model_name is None:
        model_name = MODELS["embedding"]["default"]

    # 基本設定
    config = {
        "model": model_name,
        "openai_api_key": OPENAI_API_KEY,
    }

    # 追加パラメータをマージ
    config.update(kwargs)

    return OpenAIEmbeddings(**config)


def get_embedding_dimension(model_name: str) -> int:
    """
    埋め込みモデルの次元数を取得

    Args:
        model_name: モデル名

    Returns:
        埋め込みベクトルの次元数
    """
    # OpenAIモデルの次元数マッピング
    dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,  # 旧モデル
    }

    return dimensions.get(model_name, 1536)


def calculate_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    2つの埋め込みベクトル間のコサイン類似度を計算

    Args:
        embedding1: 埋め込みベクトル1
        embedding2: 埋め込みベクトル2

    Returns:
        コサイン類似度（-1.0 〜 1.0）
    """
    # NumPy配列に変換
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # コサイン類似度の計算
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return float(cosine_sim)


def batch_embed_texts(
    texts: list[str], embeddings_model: OpenAIEmbeddings, batch_size: int = 100
) -> list[list[float]]:
    """
    テキストのリストをバッチ処理で埋め込みベクトルに変換

    Args:
        texts: テキストのリスト
        embeddings_model: 埋め込みモデル
        batch_size: バッチサイズ

    Returns:
        埋め込みベクトルのリスト
    """
    all_embeddings = []

    # バッチ処理
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def estimate_embedding_cost(
    text_length: int, model_name: str | None = None
) -> dict[str, float]:
    """
    埋め込み処理のコストを推定

    Args:
        text_length: テキストの文字数
        model_name: モデル名

    Returns:
        コスト推定情報
    """
    if model_name is None:
        model_name = MODELS["embedding"]["default"]

    # トークン数の推定（日本語の場合は文字数 * 1.5）
    estimated_tokens = int(text_length * 1.5)

    # モデル別の料金（1Kトークンあたり）
    pricing = {
        "text-embedding-3-small": 0.00002,  # $0.020 per 1M tokens
        "text-embedding-3-large": 0.00013,  # $0.130 per 1M tokens
    }

    price_per_1k = pricing.get(model_name, 0.00002)
    estimated_cost = (estimated_tokens / 1000) * price_per_1k

    return {
        "model": model_name,
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "estimated_cost_jpy": round(estimated_cost * 150, 2),  # 1USD = 150JPY想定
    }


def validate_embeddings_compatibility(existing_model: str, new_model: str) -> bool:
    """
    既存の埋め込みと新しい埋め込みの互換性を検証

    Args:
        existing_model: 既存の埋め込みモデル名
        new_model: 新しい埋め込みモデル名

    Returns:
        互換性があるかどうか
    """
    # 同じモデルなら互換性あり
    if existing_model == new_model:
        return True

    # 次元数が同じでも、異なるモデルは基本的に互換性なし
    # （ベクトル空間が異なるため）
    return False


def get_embedding_info() -> dict[str, Any]:
    """
    埋め込みモデルの情報を取得

    Returns:
        利用可能なモデルと設定情報
    """
    info = {
        "available_models": MODELS["embedding"]["options"],
        "default_model": MODELS["embedding"]["default"],
        "model_dimensions": {
            model: get_embedding_dimension(model)
            for model in MODELS["embedding"]["options"]
        },
        "recommendations": {
            "general_purpose": "text-embedding-3-small",
            "high_accuracy": "text-embedding-3-large",
            "cost_effective": "text-embedding-3-small",
        },
    }

    return info


# 将来的な拡張のための関数スタブ
def create_custom_embeddings(
    texts: list[str], custom_model_path: str
) -> list[list[float]]:
    """
    カスタム埋め込みモデルを使用（将来の拡張用）

    Args:
        texts: テキストリスト
        custom_model_path: カスタムモデルのパス

    Returns:
        埋め込みベクトル
    """
    # TODO: HuggingFaceやカスタムモデルの実装
    raise NotImplementedError("カスタム埋め込みは未実装です")


def reduce_embedding_dimension(
    embeddings: list[list[float]], target_dimension: int
) -> list[list[float]]:
    """
    埋め込みベクトルの次元削減（将来の拡張用）

    Args:
        embeddings: 元の埋め込みベクトル
        target_dimension: 削減後の次元数

    Returns:
        次元削減された埋め込みベクトル
    """
    # TODO: PCAやUMAPなどの実装
    raise NotImplementedError("次元削減は未実装です")
