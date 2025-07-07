"""
サイドバーコンポーネント
サイドバーのUI要素を管理
"""

import streamlit as st
from typing import Any
from config.settings import TEXT_SPLITTER_CONFIG, SEARCH_CONFIG, MODELS
from app.services import rag_service


def render_sidebar() -> dict[str, Any]:
    """
    サイドバーをレンダリングし、設定値を返す

    Returns:
        サイドバーの設定値を含む辞書
    """
    with st.sidebar:
        st.header("⚙️ 設定")

        st.subheader("📄 テキスト分割設定")
        chunk_size = st.slider(
            "チャンクサイズ",
            min_value=TEXT_SPLITTER_CONFIG["min_chunk_size"],
            max_value=TEXT_SPLITTER_CONFIG["max_chunk_size"],
            value=TEXT_SPLITTER_CONFIG["default_chunk_size"],
            step=100,
            help="テキストを分割する際の1チャンクあたりの文字数",
        )

        chunk_overlap = st.slider(
            "チャンク重複",
            min_value=0,
            max_value=500,
            value=TEXT_SPLITTER_CONFIG["default_chunk_overlap"],
            step=50,
            help="隣接するチャンク間で重複させる文字数",
        )

        st.subheader("🔍 検索設定")
        top_k = st.slider(
            "検索結果",
            min_value=1,
            max_value=SEARCH_CONFIG["max_top_k"],
            value=SEARCH_CONFIG["default_top_k"],
            help="類似度検索で取得する文書の数",
        )

        st.subheader("📁 ファイルアップロード")
        pdf_file = st.file_uploader(
            "コーディング規約 PDF",
            type=["pdf"],
            help="処理するPDFファイルを選択してください",
        )

        st.subheader("💾 データ管理")
        col1, col2 = st.columns(2)

        with col1:
            reset_clicked = st.button(
                "🗑️ リセット",
                use_container_width=True,
                help="ベクトルストアをリセットします",
            )

        with col2:
            if st.button("ℹ️ 情報", use_container_width=True, help="システム情報を表示"):
                st.session_state.show_info = not st.session_state.get(
                    "show_info", False
                )

        st.subheader("🤖 モデル設定")

        chat_model_info = rag_service.get_chat_model_info()
        model_options = chat_model_info["available_models"]
        model_descriptions = chat_model_info["model_descriptions"]

        current_model = st.session_state.get(
            "current_model", chat_model_info["default_model"]
        )

        model_name = st.selectbox(
            "モデル選択",
            options=model_options,
            index=(
                model_options.index(current_model)
                if current_model in model_options
                else 0
            ),
            format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("temperature", 0.0),
            step=0.1,
            help="生成の多様性（0=決定的、2=創造的）",
        )

        update_model = st.button(
            "🔄 モデル更新", use_container_width=True, help="モデル設定を適用します"
        )

        with st.expander("📖 使用方法", expanded=False):
            st.markdown(
                """
            1. **PDFアップロード**: コーディング規約のPDFを選択
            2. **チャンク設定**: 必要に応じて調整
            3. **質問入力**: メイン画面で質問を入力
            4. **回答生成**: RAGまたは検索モードを選択して実行
            
            **ヒント**: 
            - チャンクサイズが大きいほど文脈を保持しやすい
            - 重複が多いほど検索精度が向上する可能性がある
            """
            )

    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "pdf_file": pdf_file,
        "reset_clicked": reset_clicked,
        "model_name": model_name,
        "temperature": temperature,
        "update_model": update_model,
    }

    if model_name != current_model:
        st.session_state.current_model = model_name
    if temperature != st.session_state.get("temperature", 0.0):
        st.session_state.temperture = temperature

    return config
