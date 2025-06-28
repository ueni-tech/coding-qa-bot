# main.py（最小版）
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# 設定が正しく読み込めるかテスト
from config import settings

print(f"✅ 設定読み込み成功: {settings.BASE_DIR}")
print(f"✅ OpenAI APIキー: {'設定済み' if settings.OPENAI_API_KEY else '未設定'}")
