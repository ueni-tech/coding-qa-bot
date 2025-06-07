FROM python:3.11-slim

# 作業ディレクトリ設定
WORKDIR /app

# システムパッケージ更新とPopplerインストール（PDF処理用）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Poetryインストール
RUN pip install poetry

# Poetry設定（仮想環境をコンテナ内に作成しない）
RUN poetry config virtualenvs.create false

# プロジェクトファイルコピー
COPY pyproject.toml poetry.lock* ./

# 依存関係インストール（本番環境のみ）
RUN poetry install --only=main --no-root

# アプリケーションファイルコピー
COPY . .

# Streamlitポート公開
EXPOSE 8501

# Streamlitアプリ起動
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]