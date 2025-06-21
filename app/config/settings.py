"""
アプリケーション設定管理
Laravel の config/app.php のような役割
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from chromadb.config import Settings

# 環境変数の読み込み
load_dotenv()


class AppConfig:
    """アプリケーション設定クラス"""

    # ベースディレクトリ
    BASE_DIR = Path(__file__).parent.parent.parent
    PERSIST_DIR = BASE_DIR / "vectorstore"

    # OpenAI設定
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 埋め込みモデル設定
    EMBEDDING_MODEL = "text-embedding-3-small"

    # デフォルトLLMモデル
    DEFAULT_LLM_MODEL = "gpt-4o-mini"

    # 利用可能なLLMモデル一覧
    AVAILABLE_LLM_MODELS = [
        "gpt-4o-mini",  # 最もコスパに優れたモデル（推奨）
        "gpt-4.1-nano",  # 2025年時点最新、最安・最速モデル
        "gpt-4.1-mini",  # GPT-4oより83%安価で高性能
        "gpt-4o",  # バランス重視
        "gpt-4-turbo",  # 高性能モデル
        "gpt-3.5-turbo",  # 旧世代（非推奨）
    ]

    # ChromaDB設定
    CHROMA_SETTINGS = Settings(
        allow_reset=True,
        is_persistent=True,
        persist_directory=str(PERSIST_DIR),
        anonymized_telemetry=False,
    )

    # テキスト分割の初期設定
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_TOP_K = 3

    # Streamlit設定
    PAGE_CONFIG = {
        "page_title": "コーディング規約QAボット",
        "page_icon": "🤖",
        "layout": "wide",
    }

    # RAGプロンプトテンプレート
    RAG_PROMPT_TEMPLATE = """\
    あなたは優秀なコーディング規約アシスタントです。
    提供されたコンテキスト情報を基に、ユーザーの質問に正確で実用的な回答を提供してください。

    **重要な指示:**
    - コンテキストに含まれている情報のみを使用してください
    - コンテキストに情報がない場合は、「提供された規約文書に該当する情報が見つかりません」と回答してください
    - コードの例やベストプラクティスを含めて、実用的な回答を心がけてください
    - 回答は日本語で行い、わかりやすく説明してください

    **コンテキスト:**
    {context}

    **質問:**
    {question}

    **回答:**"""

    @classmethod
    def validate_config(cls) -> bool:
        """設定の妥当性チェック"""
        if not cls.OPENAI_API_KEY:
            return False
        return True

    @classmethod
    def get_model_config(cls, model_name: str = None) -> Dict[str, Any]:
        """特定のモデルの設定を取得"""
        return {
            "model": model_name or cls.DEFAULT_LLM_MODEL,
            "temperature": 0,
            "openai_api_key": cls.OPENAI_API_KEY,
        }
