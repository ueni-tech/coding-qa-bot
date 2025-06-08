import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    """
    Streamlitアプリケーションのメイン関数
    """

    # ページのmeta情報設定
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
