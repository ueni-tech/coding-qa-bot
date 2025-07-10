"""
モデル関連のユーティリティ
モデル情報の管理や共通処理をここに集約
"""

from config.settings import MODELS
from app.services import embedding_service


def get_all_model_info() -> dict:
    """
    全モデルの統合情報を取得

    Returns:
        埋め込みモデルとチャットモデルの統合情報
    """
    return {
        "embedding": {
            "available": MODELS["embedding"]["options"],
            "default": MODELS["embedding"]["default"],
            "dimensions": {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            },
        },
        "chat": {
            "available": MODELS["chat"]["options"],
            "default": MODELS["chat"]["default"],
            "descriptions": {
                "gpt-4o-mini": "最もコストパフォーマンスに優れたモデル",
                "gpt-4.1-nano": "2025年最新、最速・最安モデル",
                "gpt-4.1-mini": "GPT-4oより83%安価で高性能",
                "gpt-4o": "バランスの取れた汎用モデル",
                "gpt-4-turbo": "高性能・高精度モデル",
            },
        },
    }


# NOTE
# チャットは「トークン数」を直接指定（やり取りの内容が複雑で、正確なトークン数を使いたい場合が多い）。
def estimate_chat_cost(
    chat_model: str | None = None,
    num_questions: int = 10,
    tokens_per_question: int = 1000,
) -> dict:
    """
    チャットのコストを推定

    Args:
        chat_model: チャットモデル名
        num_questions: 想定される質問数
        tokens_per_question: 質問あたりのトークン数

    Returns:
        チャットコスト情報
    """
    if chat_model is None:
        chat_model = MODELS["chat"]["default"]

    # チャットモデルの料金（1Kトークンあたり）
    chat_pricing = {
        "gpt-4o-mini": 0.00015,  # $0.150 per 1M input tokens
        "gpt-4.1-nano": 0.00010,  # 仮定値
        "gpt-4.1-mini": 0.00012,  # 仮定値
        "gpt-4o": 0.00250,  # $2.50 per 1M input tokens
        "gpt-4-turbo": 0.01000,  # $10.00 per 1M input tokens
    }

    price_per_1k = chat_pricing.get(chat_model, 0.00015)
    cost_per_question = (tokens_per_question / 1000) * price_per_1k
    total_cost = cost_per_question * num_questions

    return {
        "model": chat_model,
        "cost_per_question_usd": round(cost_per_question, 6),
        "total_cost_usd": round(total_cost, 6),
        "total_cost_jpy": round(total_cost * 150, 2),
        "assumptions": {
            "questions": num_questions,
            "tokens_per_question": tokens_per_question,
        },
    }


def estimate_total_cost(
    text_length: int,
    embedding_model: str | None = None,
    chat_model: str | None = None,
    num_questions: int = 10,
    tokens_per_question: int = 1000,
) -> dict:
    """
    RAGシステム全体のコスト推定

    Args:
        text_length: 処理するテキストの文字数
        embedding_model: 埋め込みモデル名
        chat_model: チャットモデル名
        num_questions: 想定される質問数
        tokens_per_question: 質問あたりのトークン数

    Returns:
        総合コスト推定情報
    """
    # 埋め込みコストの計算
    embedding_cost_info = embedding_service.estimate_embedding_cost(
        text_length, embedding_model
    )

    # チャットコストの計算
    chat_cost_info = estimate_chat_cost(chat_model, num_questions, tokens_per_question)

    # 総コストを計算
    total_cost_usd = (
        embedding_cost_info["estimated_cost_usd"] + chat_cost_info["total_cost_usd"]
    )

    return {
        "embedding": {
            "model": embedding_cost_info["model"],
            "cost_usd": embedding_cost_info["estimated_cost_usd"],
            "tokens": embedding_cost_info["estimated_tokens"],
        },
        "chat": {
            "model": chat_cost_info["model"],
            "cost_usd": chat_cost_info["total_cost_usd"],
            "cost_per_question_usd": chat_cost_info["cost_per_question_usd"],
        },
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_jpy": round(total_cost_usd * 150, 2),
        "assumptions": {
            "text_length": text_length,
            "questions": num_questions,
            "tokens_per_question": tokens_per_question,
        },
    }


def validate_model_selection(
    model_name: str, model_type: str = "chat"
) -> tuple[bool, str]:
    """
    モデル選択の妥当性を検証

    Args:
        model_name: モデル名
        model_type: "chat" または "embedding"

    Returns:
        (is_valid, message) のタプル
    """
    if model_type not in ["chat", "embedding"]:
        return False, "モデルタイプは 'chat' または 'embedding' である必要があります"

    available_models = MODELS[model_type]["options"]

    if model_name not in available_models:
        return False, f"'{model_name}' は利用可能な{model_type}モデルではありません"

    warnings = {
        "gpt-4.5-preview": "高コストモデルです。本当に必要な場合のみ使用してください",
        "gpt-4o": "高コストモデルです。よりコストパフォーマンスに優れるモデルを検討してください。",
        "o3": "高コストモデルです。よりコストパフォーマンスに優れるモデルを検討してください。",
        "text-embedding-3-large": "大きな埋め込みモデルです。ストレージコストが増加します",
    }
    if model_name in warnings:
        return True, warnings[model_name]

    return True, "OK"


def get_model_recommendations(use_case: str) -> dict:
    """
    ユースケースに基づくモデル推奨

    Args:
        use_case: ユースケース（"cost", "quality", "speed"）

    Returns:
        推奨モデル情報
    """
    recommendations = {
        "cost": {
            "embedding": "text-embedding-3-small",
            "chat": "gpt-4o-mini",
            "reason": "コストを最小限に抑えながら十分な品質を確保",
        },
        "quality": {
            "embedding": "text-embedding-3-large",
            "chat": "o3",
            "reason": "高い精度と品質を重視",
        },
        "speed": {
            "embedding": "text-embedding-3-small",
            "chat": "gpt-4.1-nano",
            "reason": "レスポンス速度を最優先",
        },
        "balanced": {
            "embedding": "text-embedding-3-small",
            "chat": "o4-mini",
            "reason": "コスト、品質、速度のバランスを重視",
        },
    }

    return recommendations.get(use_case, recommendations["balanced"])
