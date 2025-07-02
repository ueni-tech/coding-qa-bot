"""
ベクトルストアリポジトリ
データ層の操作をここに集約することで、将来的にChroma以外への切り替えを容易にする
"""

import gc
import shutil
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from config.settings import VECTORSTORE_DIR, CHROMA_SETTINGS


def create_vectorstore(texts: list[str], embeddings: Embeddings) -> Chroma:
    """
    新しいベクトルストアを作成する

    Args:
        texts: 保存するテキストのリスト
        embeddings: 埋め込みモデル

    Returns:
        作成されたベクトルストア
    """
    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embeddings, client_settings=CHROMA_SETTINGS
    )

    vectorstore.persist()
    return vectorstore


def load_vectorstore(embeddins: Embeddings) -> Chroma | None:
    """
    既存のベクトルストアを読み込む

    Args:
        embeddings: 埋め込むモデル

    Returns:
        ベクトルストア（存在しない場合はNone
    """
    if not VECTORSTORE_DIR.exists():
        return None

    try:
        return Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddins,
            client_settings=CHROMA_SETTINGS,
        )
    except Exception:
        return None


def delete_vectorstore() -> bool:
    """
    ベクトルストアを削除する

    Returns:
        削除成功の可否
    """
    try:
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
        return True
    except Exception:
        return False


def reset_vectorestore(
    old_vectorstore: Chroma | None, new_texts: list[str], embeddigs: Embeddings
) -> Chroma:
    """
    ベクトルストアをリセットして再構築する

    Args:
        old_vectorstore: 既存のベクトルストア
        new_texts: 新しいベクトルストア
        embeddings: 埋め込みモデル

    Returns:
        新しいベクトルストア
    """
    if old_vectorstore:
        try:
            old_vectorstore._client.reset()
        except Exception:
            pass
        del old_vectorstore
        gc.collect()

    delete_vectorstore()

    return create_vectorstore(new_texts, embeddigs)


def get_vectorstore_info(vectorstore: Chroma) -> dict:
    """
    ベクトルストアの情報を取得する

    Args:
        vectorstore: ベクトルストア

    Returns:
        ベクトルストアの情報
    """
    try:
        collection = vectorstore._collection
        return {
            "collection_id": collection.id,
            "document_count": collection.count(),
            "persist_directory": str(VECTORSTORE_DIR),
        }
    except Exception as e:
        return {"error": str(e)}


def cleanup_old_collections(keep_uuid: str | None = None):
    """
    古いコレクションをクリーンアップする

    Args:
        keep_uuid: 保持するコレクションのUUID
    """
    if not VECTORSTORE_DIR.exists():
        return

    for path in VECTORSTORE_DIR.iterdir():
        if path.is_dir() and path.name != keep_uuid:
            try:
                shutil.rmtree(path)
            except Exception:
                pass
