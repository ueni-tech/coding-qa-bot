from dotenv import load_dotenv
from io import BytesIO  # バイト列をファイルのように扱うためのモジュール
import os

import streamlit as st
import PyPDF2  # PDF読み取り、実ファイルを使わずメモリ上にPDFなどのデータを一時的に保存して処理するライブラリ

load_dotenv()


@st.cache_data  # 関数の戻り値（PDFから抽出したテキスト）をキャッシュするためのデコレータ
def extract_text_from_pdf(pdf_file):
    """
    PDFファイルを受け取りその中からテキスト抽出する関数

    Args:
        pdf_file: アップロードされたPDFファイル

    Returns:
        str: 抽出されたテキスト
    """

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        return text

    except Exception as e:
        st.error(f"PDFの読み込みでエラーが発生しました: {str(e)}")
        return None


def main():
    """
    Streamlitアプリケーションのメイン関数
    """

    # NOTE: ページのmeta情報設定
    st.set_page_config(
        page_title="コーディング規約QAボット", page_icon="🤖", layout="wide"
    )

    st.title("🤖 コーディング規約QAボット")
    st.markdown("社内のコーディング規約について質問してください！")

    with st.sidebar:
        st.header("設定")
        st.info("OpenAI APIキーが設定されているか確認してください")

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ APIキーが設定されています")
        else:
            st.error("❌ APIキーが設定されていません")

        st.divider()

        st.subheader("📄 PDFアップロード")
        uploaded_file = st.file_uploader(
            "コーディング規約PDFをアップロード",
            type="pdf",
            help="コーディング規約が記載されたPDFファイルをアップロードしてください",
        )

        if uploaded_file is not None:
            with st.spinner("PDFを読み込み中"):
                extract_text = extract_text_from_pdf(uploaded_file)

                if extract_text:
                    st.success("✅ PDF読み込み完了")

                    # NOTE: Streamlitは再実行型のため、PDFテキストをセッションに保持して再描画時も維持
                    st.session_state.pdf_text = extract_text

                    st.info(f"文字数: {len(extract_text)}")

                    with st.expander("📖 テキストプレビュー"):
                        st.text_area(
                            "抽出されたテキスト（最初の500文字）",
                            (
                                extract_text[:500] + "..."
                                if len(extract_text) > 500
                                else extract_text
                            ),
                            height=200,
                            disabled=True,
                        )

    st.subheader("質問を入力してください")

    user_question = st.text_input(
        "質問:", placeholder="例: pythonの変数命名規則を教えて"
    )

    if st.button("質問する"):
        if user_question:
            st.info(f"質問: {user_question}")
            st.warning(
                "⚠️ まだPDF処理機能は実装されていません（次のステップで実装します）"
            )
        else:
            st.error("質問してください")


if __name__ == "__main__":
    main()
