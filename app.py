import gc
import os
import shutil
from pathlib import Path

import PyPDF2
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------
# 0. 定数・設定
# ---------------------------------------------------------------------
load_dotenv()

# IMPORTANT: 2025年6月現在の最適なモデル選択について
# - gpt-4o-mini: 128K context、視覚対応、最もコスト効率が良い
# - gpt-4.1-nano: 2025年最新、1M context、最安・最速
# - gpt-4.1-mini: GPT-4oより83%安価、レイテンシー半分
# RAGシステムには複雑な推論が不要なため、gpt-4o-miniが最適


# NOTE
# 通常"/"は割り算の演算子だがpathlibが「パス結合演算子」として再定義している
PERSIST_DIR = (
    Path(__file__).parent
    / "vectorstore"  # プロジェクト直下に"vectorstore"(/vectorstore)となる
)

# NOTE
# ChromaDB用の設定。この設定を使ってLangchainのChromaベクトルストアを初期化・永続化する
CHROMA_SETTINGS = Settings(
    allow_reset=True,
    is_persistent=True,
    persist_directory=str(PERSIST_DIR),
    anonymized_telemetry=False,
)

# RAG用プロンプトテンプレート
RAG_PROMPT_TEMPLATE = """\
あなたは優秀なコーディング規約アシスタントです。
提供されたコンテキスト情報を基に、ユーザーの質問に正確で実用的な回答を提供してください。

**重要な指示:**
- コンテキストに含まれている情報のみを使用してください
- コンテキストに情報がない場合は、「提供された規約文書に該当する情報が見つかりません」と回答してください
- コードの例やベストプラクティスを含めて、実用的な回答を心がけてください
- 回答は日本語で行い、わかりやすく説明してください

**コンテキスト:**
{context}

**質問:**
{question}

**回答:**"""


# ---------------------------------------------------------------------
# 1. ユーティリティ
# ---------------------------------------------------------------------
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


@st.cache_data
def split_text(text: str, size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def count_tokens(text: str, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


@st.cache_resource
def init_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


@st.cache_resource
def init_llm():
    """ChatOpenAIモデルを初期化する関数"""
    return ChatOpenAI(
        model="gpt-4o-mini",  # 2025年現在、最もコスパに優れたモデル
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def format_docs(docs):
    """ドキュメントをフォーマットしてコンテキストとして使用する関数"""
    return "\n\n".join(doc.page_content for doc in docs)


# TODO RunnableとLCELの基礎について調べる
# https://claude.ai/share/0cd33d5c-2d8c-4520-b17a-f0c89cf42581
def create_rag_chain(vectorstore, llm):
    """RAGチェーンを作成する関数"""
    retriever = vectorstore.as_retriever(
        search_type="simikarity", search_kwargs={"k": 3}
    )

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # NOTE
    # "|"演算子はオーバーロードされている
    # LangChain の Runnable クラスでオーバーロードされており、Linuxのように左から右へデータを流すパイプラインを表現する
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ---------------------------------------------------------------------
# 2. ベクトルストア 操作
# ---------------------------------------------------------------------
def cleanup_dirs(keep_uuid: str | None):
    """
    persist_directory 配下でkeep_uuid以外のフォルダを削除するための関数

    Args:
        keep_uuid(str | None): 残したいベクトルデータの入ったフォルダ名

    Return:
        None
    """
    if not PERSIST_DIR.exists():
        return
    for p in PERSIST_DIR.iterdir():
        if p.is_dir() and p.name != keep_uuid:
            shutil.rmtree(p)


def rebuild_vectorstore(chunks, embeds):
    """
    ベクトルストアの既存接続をリセットして再構築するための関数
    1) streamlitで登録されている既存セッションの接続クローズ / reset
    2) 新規セッションの接続作成
    3) 古いuuidフォルダ削除
    """
    # 1. 既存接続を安全に閉じる
    vs_old = st.session_state.pop("vectorstore", None)
    if vs_old:
        try:
            vs_old._client.reset()
        except Exception:
            pass
        del vs_old
        gc.collect()

    # 2. 新しいベクトルストアを生成
    vs_new = Chroma.from_texts(
        texts=chunks, embedding=embeds, client_settings=CHROMA_SETTINGS
    )
    vs_new.persist()
    st.session_state.vectorstore = vs_new

    # 3. 古くなったuuidフォルダを削除
    current_uuid = vs_new._collection.id
    cleanup_dirs(current_uuid)

    return vs_new


def load_vectorstore(embeds):
    if PERSIST_DIR.exists():
        return Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeds,
            client_settings=CHROMA_SETTINGS,
        )
    return None


# ---------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="コーディング規約QAボット", page_icon="🤖", layout="wide"
    )
    st.title("🤖 コーディング規約QAボット")

    embeds = init_embeddings()
    llm = init_llm()

    if not embeds or not llm:
        st.error("OpenAI APIキーが設定されていないか、モデルの初期化に失敗しました。")
        st.info("`.env`ファイルに`OPENAI_API_KEY`を設定してください。")
        st.stop()

    # 既存ベクトルストアの読み込み
    if vs := load_vectorstore(embeds):
        st.session_state.vectorstore = vs
        st.success("✅ 既存ベクトルストアを読み込みました")

    # --- サイドバー ----------------------------------------------
    with st.sidebar:
        st.header("設定")
        pdf = st.file_uploader("📄 規約 PDF", type="pdf")
        size = st.slider("チャンクサイズ", 500, 2000, 1000, 100)
        over = st.slider("チャンク重複", 0, 500, 200, 50)

        st.subheader("検索設定")
        top_k = st.slider("検索件数", 1, 10, 3)

        st.subheader("LLM設定")
        model_choice = st.selectbox(
            "モデル選択",
            [
                "gpt-4o-mini",  # 最もコスパに優れたモデル（推奨）
                "gpt-4.1-nano",  # 2025年最新、最安・最速モデル
                "gpt-4.1-mini",  # GPT-4oより83%安価で高性能
                "gpt-4o",  # バランス重視
                "gpt-4-turbo",  # 高性能モデル
                "gpt-3.5-turbo",  # 旧世代（非推奨）
            ],
            index=0,
            help="gpt-4o-miniが最もコストパフォーマンスに優れています",
        )
        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            0.0,
            0.1,
            help="Temperatureは生成される回答の多様性を制御するパラメータです。値が高いほどランダム性が増し、低いほど決定的な（安定した）回答になります。通常は0〜1の範囲で調整します。",
        )

        if st.button("🔄 モデル設定を更新"):
            # キャッシュをクリアして新しいモデルを作成
            if "llm" in st.session_state:
                del st.session_state.llm

            st.session_state.llm = ChatOpenAI(
                model=model_choice,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

            # RAGチェーンも再構築が必要
            if "rag_chain" in st.session_state:
                del st.session_state.rag_chain
            if "retriever" in st.session_state:
                del st.session_state.retriever

            st.success(f"✅ モデル設定を更新しました: {model_choice}")

        if st.button("💣 ベクトルストアをリセット"):
            if "vectorstore" in st.session_state:
                uuid_now = st.session_state.vectorstore._collection.id
                st.session_state.vectorstore._client.reset()
                cleanup_dirs(uuid_now)
                del st.session_state.vectorstore
                if "rag_chain" in st.session_state:
                    del st.session_state.rag_chain
                if "retriever" in st.session_state:
                    del st.session_state.retriever
            st.success("ベクトルストアをリセットしました")
            st.rerun()

    # --- pdf アップロード処理 ----------------------------------------------
    if pdf:
        with st.spinner("PDF 解析中..."):
            text = extract_text_from_pdf(pdf)
            chunks = split_text(text, size, over)
            rebuild_vectorstore(chunks, embeds)
            st.success(
                f"✅ ベクトルストア構築完了 | 文字数 {len(text):,} | "
                f"チャンク {len(chunks)} | 推定トークン {count_tokens(text):,}"
            )

    # --- 質問 UI ----------------------------------------------
    st.subheader("質問")
    if "vectorstore" not in st.session_state:
        st.warning("まずコーディング規約PDFをアップロードしてください")
        return

    q = st.text_input("質問を入力", placeholder="例: Python の変数命名規則は？")
    st.caption(f"💡 選択中のLLM: {model_choice}")
    if st.button("検索"):
        if not q:
            st.warning("質問を入力してください")
            return
        with st.spinner("検索中…"):
            res = st.session_state.vectorstore.similarity_search_with_score(q, k=top_k)
        if not res:
            st.info("関連文書が見つかりません")
            return
        for i, (doc, score) in enumerate(res, 1):
            with st.expander(f"{i}. score {score:.3f}"):
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
