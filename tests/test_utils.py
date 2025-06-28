import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# test_utils.py（動作確認用）
from app.utils import text_utils

# テキスト分割のテスト
test_text = "これはテストです。" * 100
chunks = text_utils.split_text(test_text, chunk_size=50, chunk_overlap=10)
print(f"✅ 文字数: {len(test_text)}")
print(f"✅ チャンク数: {len(chunks)}")
print(f"✅ トークン数: {text_utils.count_tokens(test_text)}")
