"""
ベクトルストアリポジトリ
データ層の操作をここに集約することで、将来的にChroma以外への切り替えを容易にする
"""

import gc
import os
import shutil
import time
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


def load_vectorstore(embeddings: Embeddings) -> Chroma | None:
    """
    既存のベクトルストアを読み込む

    Args:
        embeddings: 埋め込みモデル

    Returns:
        ベクトルストア（存在しない場合はNone）
    """
    if not VECTORSTORE_DIR.exists():
        return None

    try:
        return Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
    except Exception as e:
        print(f"ベクトルストア読み込みエラー: {e}")
        return None


def delete_vectorstore_safely() -> bool:
    """
    ベクトルストアを安全に削除する

    Returns:
        削除成功の可否
    """
    try:
        if VECTORSTORE_DIR.exists():
            # ファイルロックを回避するため少し待機
            time.sleep(0.1)
            shutil.rmtree(VECTORSTORE_DIR)
            # 削除後に少し待機
            time.sleep(0.1)
        return True
    except Exception as e:
        print(f"ベクトルストア削除エラー: {e}")
        return False


def reset_vectorstore(
    old_vectorstore: Chroma | None, new_texts: list[str], embeddings: Embeddings
) -> Chroma:
    """
    ベクトルストアをリセットして再構築する

    Args:
        old_vectorstore: 既存のベクトルストア
        new_texts: 新しいテキストリスト
        embeddings: 埋め込みモデル

    Returns:
        新しいベクトルストア
    """
    # 既存接続を安全に閉じる
    if old_vectorstore:
        try:
            old_vectorstore._client.reset()
        except Exception as e:
            print(f"既存接続のリセットエラー: {e}")

        # 参照を削除してガベージコレクション
        del old_vectorstore
        gc.collect()
        time.sleep(0.2)  # 少し待機

    # 古いデータを削除
    old_uuid = None
    try:
        # 既存のコレクションUUIDを保存（クリーンアップ用）
        if VECTORSTORE_DIR.exists():
            for path in VECTORSTORE_DIR.iterdir():
                if path.is_dir():
                    old_uuid = path.name
                    break
    except Exception:
        pass

    # 新しいベクトルストアを作成
    new_vectorstore = create_vectorstore(new_texts, embeddings)

    # 古いコレクションをクリーンアップ
    if old_uuid and new_vectorstore:
        try:
            new_uuid = new_vectorstore._collection.id
            cleanup_old_collections(new_uuid)
        except Exception as e:
            print(f"クリーンアップエラー: {e}")

    return new_vectorstore


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
            "status": "正常",
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "エラー",
            "persist_directory": str(VECTORSTORE_DIR),
        }


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
                time.sleep(0.1)  # ファイルロックを回避
                shutil.rmtree(path)
                print(f"古いコレクションを削除しました: {path.name}")
            except Exception as e:
                print(f"コレクション削除エラー ({path.name}): {e}")


def check_vectorstore_permissions() -> dict:
    """
    ベクトルストアディレクトリの権限をチェックする

    Returns:
        権限チェックの結果
    """
    result = {
        "directory_exists": VECTORSTORE_DIR.exists(),
        "is_writable": False,
        "is_readable": False,
        "path": str(VECTORSTORE_DIR),
    }

    try:
        # 読み込み権限チェック
        result["is_readable"] = os.access(VECTORSTORE_DIR, os.R_OK)

        # 書き込み権限チェック
        result["is_writable"] = os.access(VECTORSTORE_DIR, os.W_OK)

        # 実際にテストファイルを作成してみる
        test_file = VECTORSTORE_DIR / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()  # 削除
            result["write_test"] = "成功"
        except Exception as e:
            result["write_test"] = f"失敗: {e}"

    except Exception as e:
        result["error"] = str(e)

    return result
