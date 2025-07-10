"""
アプリケーションのエントリーポイント
Laravelのpublic/index.phpに相当
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
# これにより、どこからでもモジュールをインポートできるようになる
sys.path.append(str(Path(__file__).parent))

# メインアプリケーションの起動
from views.components.main import main

if __name__ == "__main__":
    main()
