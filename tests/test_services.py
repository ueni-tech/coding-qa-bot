from app.services import pdf_service, embedding_service

# PDF処理のテスト
with open("test.pdf", "rb") as f:
    is_valid, msg = pdf_service.validate_pdf(f)
    print(f"✅ PDF検証: {msg}")

# 埋め込みモデルのテスト
embeddings = embedding_service.create_embeddings()
print(f"✅ 埋め込みモデル作成成功")
