import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import tempfile
import shutil

from app.repositories import vectorstore_repository
from app.services import embedding_service


def test_ベクトルストアのパスが正しい():
    """設定されたパスが存在するか"""
    from config.settings import VECTORSTORE_DIR

    assert VECTORSTORE_DIR.exists()
    print(f"✅ ベクトルストアパス: {VECTORSTORE_DIR}")


def test_ベクトルストアの作成と削除():
    """ベクトルストアの基本操作"""
    # 一時ディレクトリでテスト
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # テスト用のテキスト
        texts = ["テスト文書1", "テスト文書2", "テスト文書3"]

        # 埋め込みモデル（モックで代用）
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 1536 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 1536

        embeddings = MockEmbeddings()

        # ベクトルストア作成
        vs = vectorstore_repository.create_vectorstore(texts, embeddings)
        assert vs is not None
        print("✅ ベクトルストア作成成功")

        # 削除
        success = vectorstore_repository.delete_vectorstore()
        print(f"✅ ベクトルストア削除: {success}")

    finally:
        # クリーンアップ
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
