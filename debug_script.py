"""
ベクトルストア問題のデバッグスクリプト
"""

from config.settings import VECTORSTORE_DIR
from vectorstore_repository import check_vectorstore_permissions
import os


def debug_vectorstore_issue():
    """ベクトルストア問題をデバッグする"""

    print("=== ベクトルストア問題デバッグ ===\n")

    # 1. パス情報
    print(f"1. ベクトルストアパス: {VECTORSTORE_DIR}")
    print(f"   - 絶対パス: {VECTORSTORE_DIR.absolute()}")
    print(f"   - 存在確認: {VECTORSTORE_DIR.exists()}")

    # 2. 権限チェック
    print("\n2. 権限チェック:")
    permissions = check_vectorstore_permissions()
    for key, value in permissions.items():
        print(f"   - {key}: {value}")

    # 3. 親ディレクトリの権限
    print(f"\n3. 親ディレクトリの権限:")
    parent_dir = VECTORSTORE_DIR.parent
    print(f"   - パス: {parent_dir}")
    print(f"   - 存在: {parent_dir.exists()}")
    print(f"   - 読み込み可能: {os.access(parent_dir, os.R_OK)}")
    print(f"   - 書き込み可能: {os.access(parent_dir, os.W_OK)}")

    # 4. 既存ファイル確認
    print(f"\n4. 既存ファイル確認:")
    if VECTORSTORE_DIR.exists():
        try:
            files = list(VECTORSTORE_DIR.iterdir())
            print(f"   - ファイル数: {len(files)}")
            for file in files[:5]:  # 最初の5個まで表示
                print(
                    f"   - {file.name}: {'ディレクトリ' if file.is_dir() else 'ファイル'}"
                )
        except Exception as e:
            print(f"   - ファイル一覧取得エラー: {e}")
    else:
        print("   - ディレクトリが存在しません")

    # 5. 環境情報
    print(f"\n5. 環境情報:")
    print(f"   - OS: {os.name}")
    print(f"   - ユーザーID: {os.getuid() if hasattr(os, 'getuid') else 'N/A'}")
    print(f"   - 現在のディレクトリ: {os.getcwd()}")


if __name__ == "__main__":
    debug_vectorstore_issue()
