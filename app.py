import os
import shutil
import PyPDF2

import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

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


@st.cache_data
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    受け取ったテキストを指定サイズのチャンクに分割する関数

    Args:
        text(str): 分割対象テキスト
        chunk_size(int): 各チャンクの最大文字数
        chunk_overlap(int): チャンク間の重複文字数

    Returns:
        list: 分割されたテキストチャンクのリスト
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # len(文字数)ベースでチャンクサイズを指定
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(text)

        return chunks

    except Exception as e:
        st.error(f"テキスト分割でエラーが発生しました: {str(e)}")
        return []


def count_tokens(text, model="gpt-3.5-turbo"):
    """
    テキストのトークン数をカウントする関数

    Args:
        text(str): カウント対象のテキスト
        model(str): 使用するモデル

    Returns:
        int: トークン数
    """
    # TODO
    # なぜカウントする関数が必要？
    # → API制限、コスト見積、文脈制御のため

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        st.error(f"トークンカウントでエラーが発生しました: {str(e)}")
        return 0


@st.cache_resource
def initialize_embeddings():
    """Open AI Embeddingsを初期化する関数(@st.cache_resourceでキャッシュ)"""
    try:
        # NOTE
        # api_keyには型SecretStr | Noneが要求されているがos.getenv("OPEN_API_KEY")でいいみたい
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        return embeddings
    except Exception as e:
        st.error(f"Embedding初期化でエラーが発生しました: {str(e)}")
        return None


def create_vector_store(text_chunks, embeddings, persist_directory="./vectorstore"):
    """
    ChromaDBベクトルストアを作成する関数

    Args:
        text_chunks(list): テキストチャンクのリスト
        embeddings: OpenAIEmbeddingsのオブジェクト
        persist_directory(str): データ永続化ディレクトリへのパス

    Return:
        Chroma: 作成されたベクトルストア
    """
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        vectorstore = Chroma.from_texts(
            texts=text_chunks, embedding=embeddings, persist_directory=persist_directory
        )

        vectorstore.persist()

        return vectorstore

    except Exception as e:
        st.error(f"ベクトルストア作成でエラーが発生しました: {str(e)}")
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

        st.subheader("⚙️ テキスト分割設定")
        chunk_size = st.slider(
            "チャンクサイズ",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="各テキストチャンクの最大文字数",
        )

        chunk_overlap = st.slider(
            "チャンク重複",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="隣接するチャンクの重複文字数",
        )

        if uploaded_file is not None:
            with st.spinner("PDFを読み込み中"):
                extract_text = extract_text_from_pdf(uploaded_file)

                if extract_text:
                    text_chunks = split_text_into_chunks(
                        extract_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

                    if text_chunks:
                        st.success("✅ PDF読み込み完了")

                        # NOTE: Streamlitは再実行型のため、PDFテキストやchunkテキストをセッションに保持して再描画時も維持
                        st.session_state.pdf_text = extract_text
                        st.session_state.text_chunks = text_chunks

                        st.info(
                            f"""
                        📊 **処理結果:**
                        - 総文字数: {len(extract_text):,}
                        - チャンク数: {len(text_chunks)}
                        - 推定トークン数: {count_tokens(extract_text):,}
                                """
                        )

                        with st.expander("📖 チャンクプレビュー"):
                            for i, chunk in enumerate(text_chunks[:3]):
                                st.write(f"**チャンク {i+1}**")
                                st.text_area(
                                    f"チャンク{i+1}の内容",
                                    chunk[:300] + "..." if len(chunk) > 300 else chunk,
                                    height=100,
                                    disabled=True,
                                    key=f"chunk_{i}",
                                )

                            if len(text_chunks) > 3:
                                st.info(f"... 他 {len(text_chunks) - 3} 個のチャンク")

    st.subheader("質問を入力してください")

    if "text_chunks" not in st.session_state:
        st.warning("📄 まずサイドバーからコーディング規約PDFをアップロードしてください")
        return

    embeddings = initialize_embeddings()
    if not embeddings:
        st.error("❌ Embeddings初期化に失敗しました。APIキーを確認してください。")
        return

    user_question = st.text_input(
        "質問:", placeholder="例: pythonの変数命名規則を教えて"
    )

    if st.button("質問する"):
        if user_question:
            st.info(f"質問: {user_question}")

            with st.spinner("ベクトル検索を実施中..."):
                try:
                    # NOTE: 質問文をベクトル化
                    question_embeddig = embeddings.embed_query(user_question)

                    # NOTE: （開発中）ベクトル化対象のチャンクを先頭10個に制限
                    chunks = st.session_state.text_chunks[:10]
                    # NOTE: PDFからチャンク化したテキストをベクトル化
                    chunk_embeddings = embeddings.embed_documents(chunks)

                    st.success("✅ ベクトル化完了")
                    st.info(f"検索対象チャンク数: {len(chunks)}")
                    st.warning(
                        "⚠️ まだベクトルデータベースは実装されていません（次のステップで実装します）"
                    )

                except Exception as e:
                    st.error(f"ベクトル化でエラーが発生しました: {str(e)}")

        else:
            st.error("質問してください")


if __name__ == "__main__":
    main()
