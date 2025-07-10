"""
メインUI
UIロジックのみに専念し、ビジネスロジックはコントローラーに委譲
"""

import streamlit as st
from app.controllers import rag_controller
from config.settings import UI_CONFIG, TEXT_SPLITTER_CONFIG, SEARCH_CONFIG
from config.prompts import SUCCESS_MESSAGES, ERROR_MESSAGES
from views.components import sidebar, chat_interface


def main():
    """メインアプリケーション"""
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout=UI_CONFIG["layout"],
    )

    st.title(f"{UI_CONFIG['page_icon']} {UI_CONFIG['page_title']}")

    init_result = rag_controller.initialize_system()

    if not init_result["success"]:
        st.error(init_result["error"])
        if "instruction" in init_result:
            st.info(init_result["instruction"])
        st.stop()

    embeddings = init_result["embeddings"]
    llm = init_result.get("llm")

    if init_result.get("vectorstore_loaded"):
        st.session_state.vectorstore = init_result["vectorstore"]
        st.success(SUCCESS_MESSAGES["vectorstore_loaded"])

    sidebar_config = sidebar.render_sidebar()

    if sidebar_config["pdf_file"]:
        with st.spinner("PDF 解析中"):
            result = rag_controller.process_pdf_upload(
                sidebar_config["pdf_file"],
                sidebar_config["chunk_size"],
                sidebar_config["chunk_overlap"],
                embeddings,
            )

            if result["success"]:
                stats = result["stats"]
                st.success(
                    SUCCESS_MESSAGES["vectorstore_built"].format(
                        chars=stats["text_length"],
                        chunks=stats["chunk_count"],
                        tokens=stats["token_count"],
                    )
                )
            else:
                st.error(f"エラー: {result['error']}")

    if sidebar_config["reset_clicked"]:
        if rag_controller.reset_system():
            st.success(SUCCESS_MESSAGES["vectorstore_reset"])
            st.rerun()
        else:
            st.error("リセットに失敗しました")

    if sidebar_config["update_model"]:
        result = rag_controller.update_model_settings(
            sidebar_config["model_name"], sidebar_config["temperature"]
        )
        if result["success"]:
            st.success(SUCCESS_MESSAGES["model_updated"].format(model=result["model"]))
        else:
            st.error(f"モデル更新エラー: {result['error']}")

    chat_interface.render_chat_interface(
        embeddings=embeddings, llm=llm, top_k=sidebar_config["top_k"]
    )


if __name__ == "__main__":
    main()
