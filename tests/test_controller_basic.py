import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.controllers import rag_controller


def test_システム初期化の基本動作():
    """initialize_system関数の基本テスト"""

    result = rag_controller.initialize_system()

    if result["success"]:
        print("✅ システム初期化成功")
        assert "embeddings" in result
        assert "llm" in result
    else:
        print(f"⚠️ システム初期化失敗: {result.get('error', '')}")
        assert "error" in result


def test_モデル設定更新の動作():
    """update_model_settings関数のテスト"""

    # StreamlitのセッションStateをモック
    class MockSessionState:
        def __init__(self):
            self.data = {}

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data.get(key)

        def __contains__(self, key):
            return key in self.data

        def __delitem__(self, key):
            if key in self.data:
                del self.data[key]

    # Streamlitをモック
    import streamlit as st

    st.session_state = MockSessionState()

    # APIキーがある場合のみテスト
    from app.services import rag_service
    from config.settings import validate_api_key

    if validate_api_key():
        result = rag_controller.update_model_settings("gpt-4o-mini", 0.5)
        assert "success" in result
        print(f"✅ モデル設定更新: {result}")


def test_process_pdf_upload_バリデーションエラー():
    """PDFバリデーション失敗時のテスト"""

    # ダミーのPDFファイルオブジェクト
    class DummyPDF:
        pass

    # pdf_service.validate_pdf をモックして失敗を返す
    from app.services import pdf_service

    pdf_service.validate_pdf = lambda pdf: (False, "不正なPDFです")

    result = rag_controller.process_pdf_upload(DummyPDF(), 1000, 200, None)
    assert not result["success"]
    assert "error" in result
    print("✅ PDFバリデーション失敗時のエラー検出")


def test_search_and_answer_search_only():
    """search_and_answerのsearch_onlyモードのテスト"""
    # ダミーのvectorstoreとllm
    dummy_vectorstore = object()
    dummy_llm = object()

    # rag_service.search_documents をモック
    from app.services import rag_service

    rag_service.search_documents = lambda vs, q, k: ["doc1", "doc2"]

    result = rag_controller.search_and_answer(
        "テスト質問", dummy_vectorstore, dummy_llm, 2, mode="search_only"
    )
    assert result["success"]
    assert result["mode"] == "search_only"
    assert result["count"] == 2
    print("✅ search_onlyモードのテスト成功")
