"""
チャットインターフェースコンポーネント
メインのチャットUIを管理
"""

import streamlit as st
from typing import Any
from app.controllers import rag_controller
from config.prompts import ERROR_MESSAGES
from views.components.sidebar import render_system_info


def render_chat_interface(embeddings: Any, llm: Any, top_k: int):
    """
    チャットインターフェースをレンダリング

    Args:
        embeddings: 埋め込みモデル
        llm: LLMモデル
        top_k: 検索件数
    """
    st.subheader("💬 質問を入力してください")

    if "vectorstore" not in st.session_state:
        _render_empty_state()
        return

    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "質問",
            placeholder="例: クラスの命名規則について教えてください",
            key="question_input",
            label_visibility="collapsed",
        )

    with col2:
        mode = st.radio(
            "モード",
            options=["🤖 RAG回答", "🔍 検索のみ"],
            index=0,
            key="search_mode",
            label_visibility="collapsed",
        )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        execute_button = st.button(
            "実行", type="primary", use_container_width=True, disabled=not question
        )

    if execute_button and question:
        _process_question(
            question=question,
            embeddings=embeddings,
            llm=llm,
            top_k=top_k,
            mode="search_only" if "検索のみ" in mode else "rag",
        )

    if st.session_state.get("show_history", False):
        _render_history()


def _render_empty_state():
    """ベクトルストアが存在しない場合の空状態表示"""
    st.warning(ERROR_MESSAGES["no_pdf"])

    with st.expander("🚀 クイックスタートガイド", expanded=True):
        st.markdown(
            """
        ### 使用方法
        
        1. **📁 PDFファイルを準備**
            - コーディング規約のPDFファイルを用意します
        
        2. **⬆️ アップロード**
            - 左サイドバーから「コーディング規約 PDF」にファイルを選択
        
        3. **⚙️ 設定調整**（オプション）
            - チャンクサイズ: テキストの分割単位
            - 検索件数: 回答生成時に参照する文書数
        
        4. **💬 質問する**
            - テキストボックスに質問を入力
            - 「実行」ボタンをクリック
        
        ### 質問例
        - 「変数の命名規則は？」
        - 「関数のコメントはどのように書くべき？」
        - 「エラーハンドリングのベストプラクティスは？」
        """
        )


def _process_question(question: str, embeddings: Any, llm: Any, top_k: int, mode: str):
    """
    質問を処理する統合関数

    Args:
        question: 質問文
        embeddings: 埋め込みモデル
        llm: LLMモデル
        top_k: 検索件数
        mode: "rag" または "search_only"
    """
    if not question.strip():
        st.warning(ERROR_MESSAGES["no_question"])
        return

    with st.spinner("🔄 処理中..." if mode == "rag" else "🔍 検索中..."):
        result = rag_controller.search_and_answer(
            question=question,
            vectorstore=st.session_state.vectorstore,
            llm=llm or st.session_state.get("llm"),
            top_k=top_k,
            mode=mode,
        )

    if result["success"]:
        _display_results(result, question)
        _add_to_history(question, result)
    else:
        _display_error(result["error"])


def _display_results(result: dict, question: str):
    """
    検索/回答結果を表示

    Args:
        result: 処理結果
        question: 元の質問
    """
    if result["mode"] == "rag":
        st.markdown("### 🤖 AI回答")

        with st.container():
            st.markdown(result["answer"])

        _display_reference_documents(result["documents"])

    else:
        _display_search_results(result["documents"])


def _display_reference_documents(documents: list):
    """参照文書を表示"""
    if not documents:
        return

    with st.expander(f"📚 参照した文書 ({len(documents)}件)", expanded=False):
        for i, doc in enumerate(documents, 1):
            st.markdown(f"**文書 {i}**")

            # テキストを見やすく表示（500文字で切り詰め）
            content = doc.page_content
            display_content = content[:500] + "..." if len(content) > 500 else content

            with st.container():
                st.text(display_content)

            if i < len(documents):
                st.divider()


def _display_search_results(documents: list):
    """検索結果を表示"""
    st.markdown("### 🔍 検索結果")

    if not documents:
        st.info(ERROR_MESSAGES["no_results"])
        return

    st.info(f"📄 {len(documents)}件の関連文書が見つかりました")

    for i, doc in enumerate(documents, 1):
        with st.expander(f"📄 文書 {i}", expanded=(i == 1)):
            st.text(doc.page_content)


def _display_error(error_message: str):
    """エラーを表示"""
    st.error(f"❌ エラー: {error_message}")

    with st.expander("🆘 トラブルシューティング", expanded=False):
        st.markdown(
            """
        ### よくあるエラーと対処法
        
        **1. API関連のエラー**
        - OpenAI APIキーが正しく設定されているか確認
        - APIの利用制限に達していないか確認
        - ネットワーク接続を確認
        
        **2. ベクトルストア関連のエラー**
        - PDFが正しくアップロードされているか確認
        - 「リセット」ボタンで初期化してから再試行
        
        **3. モデル関連のエラー**
        - 選択したモデルが利用可能か確認
        - モデル設定を更新してから再試行
        
        **それでも解決しない場合**
        - ブラウザをリロード
        - アプリケーションを再起動
        """
        )


def _add_to_history(question: str, result: dict):
    """
    質問と結果を履歴に追加

    Args:
        question: 質問
        result: 結果
    """
    if "history" not in st.session_state:
        st.session_state.history = []

    history_entry = {
        "question": question,
        "mode": result["mode"],
        "answer": result.get("answer", ""),
        "document_count": len(result.get("documents", [])),
    }

    st.session_state.history.append(history_entry)

    # 履歴は最新10件まで保持
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]


def _render_history():
    """質問履歴を表示"""
    if "history" not in st.session_state or not st.session_state.history:
        return

    with st.expander("📜 質問履歴", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.history)):
            history_index = len(st.session_state.history) - i
            st.markdown(f"**Q{history_index}**: {entry['question']}")

            if entry["mode"] == "rag" and entry["answer"]:
                # 回答を200文字で切り詰めて表示
                truncated_answer = (
                    entry["answer"][:200] + "..."
                    if len(entry["answer"]) > 200
                    else entry["answer"]
                )
                st.markdown(f"**A**: {truncated_answer}")

            st.caption(f"参照文書: {entry['document_count']}件")
            st.divider()
