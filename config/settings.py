"""
設定ファイル - LaravelのconfigディレクトリのPython版
環境変数や定数をここに集約することで、変更に強い設計にする
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from chromadb.config import Settings

# 環境変数の読み込み
load_dotenv()

# ===== パス設定 =====
# Laravelの storage_path() に相当
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
VECTORSTORE_DIR = STORAGE_DIR / "vectorstore"

# ディレクトリの作成（存在しない場合）
STORAGE_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ===== API設定 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# APIキー検証関数（共通化）
def validate_api_key() -> bool:
    """
    APIキーの妥当性を検証する

    Returns:
        APIキーが有効かどうか
    """
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())


# ===== モデル設定 =====
# 将来的にFastAPIでも使える設定構造
MODELS = {
    "embedding": {
        "default": "text-embedding-3-small",
        "options": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "chat": {
        "default": "gpt-4o-mini",
        "options": [
            "gpt-4o-mini",  # 推奨: コスパ最良
            "gpt-4.1-nano",  # 最速・最安
            "gpt-4.1-mini",  # 高性能・低価格
            "gpt-4o",  # バランス型
            "gpt-4-turbo",  # 高性能
        ],
    },
}

# ===== ChromaDB設定 =====
CHROMA_SETTINGS = Settings(
    allow_reset=True,
    is_persistent=True,
    persist_directory=str(VECTORSTORE_DIR),
    anonymized_telemetry=False,
)

# ===== テキスト処理設定 =====
TEXT_SPLITTER_CONFIG = {
    "default_chunk_size": 1000,
    "default_chunk_overlap": 200,
    "min_chunk_size": 500,
    "max_chunk_size": 2000,
}

# ===== 検索設定 =====
SEARCH_CONFIG = {
    "default_top_k": 3,
    "max_top_k": 10,
}

# ===== UI設定 =====
UI_CONFIG = {
    "page_title": "コーディング規約QAボット",
    "page_icon": "🤖",
    "layout": "wide",
}
