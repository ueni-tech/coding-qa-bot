"""
PDF処理サービス
PDFに関する処理をここに集約
"""

import PyPDF2
from typing import BinaryIO


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """
    PDFファイルからテキストを抽出する

    Args:
        pdf_file: PDFファイルオブジェクト

    Returns:
        抽出されたテキスト

    Raises:
        Exception: PDF読み込みエラー
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n".join(text_parts)

    except Exception as e:
        raise Exception(f"PDF読み込みエラー: {str(e)}")


def get_pdf_info(pdf_file: BinaryIO) -> dict:
    """
    PDFファイルの情報を取得する

    Args:
        pdf_file: PDFファイルオブジェクト

    Returns:
        PDFの情報（ページ数、メタデータなど）
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file)

        info = {
            "page_count": len(reader.pages),
            "metadata": {},
            "is_encrypted": reader.is_encrypted,
        }

        if reader.metadata:
            info["metadata"] = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }

        return info

    except Exception as e:
        return {"error": str(e)}


def validate_pdf(pdf_file: BinaryIO) -> tuple[bool, str]:
    """
    PDFファイルの妥当性を検証する

    Args:
        pdf_file: PDFファイルオブジェクト

    Returns:
        (is_valid, message) のタプル
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file)

        if reader.is_encrypted:
            return False, "暗号化されたPDFは処理できません"

        if len(reader.pages) == 0:
            return False, "PDFにページが含まれていません"

        first_page_text = reader.pages[0].extract_text()
        if not first_page_text or len(first_page_text.strip()) == 0:
            return False, "PDFからテキストを抽出できません"

        return True, "OK"

    except Exception as e:
        return False, f"PDFの検証中にエラーが発生しました: {str(e)}"
