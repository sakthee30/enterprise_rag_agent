# ingestion/loader.py

import os
import docx2txt
import pypdf
from typing import Optional


def load_pdf(file_path: str) -> Optional[str]:
    """
    Reads a PDF file and returns all text as a single string.
    """
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"✅ Loaded PDF: {os.path.basename(file_path)} ({len(text)} characters)")
        return text
    except Exception as e:
        print(f"❌ Failed to load PDF {file_path}: {e}")
        return None


def load_docx(file_path: str) -> Optional[str]:
    """
    Reads a DOCX file and returns all text as a single string.
    """
    try:
        text = docx2txt.process(file_path)
        print(f"✅ Loaded DOCX: {os.path.basename(file_path)} ({len(text)} characters)")
        return text
    except Exception as e:
        print(f"❌ Failed to load DOCX {file_path}: {e}")
        return None


def load_txt(file_path: str) -> Optional[str]:
    """
    Reads a plain text file and returns content as string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ Loaded TXT: {os.path.basename(file_path)} ({len(text)} characters)")
        return text
    except Exception as e:
        print(f"❌ Failed to load TXT {file_path}: {e}")
        return None


def load_document(file_path: str) -> Optional[str]:
    """
    Master loader — detects file type and calls the right loader.
    Supports: .pdf, .docx, .txt
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}. Supported: .pdf, .docx, .txt")
        return None


def load_all_documents(data_folder: str = "./data") -> dict:
    """
    Loads ALL documents from the /data folder.
    Returns a dict: { filename: text_content }
    """
    documents = {}

    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        return documents

    files = [f for f in os.listdir(data_folder)
             if f.endswith((".pdf", ".docx", ".txt"))]

    if not files:
        print(f"⚠️  No PDF, DOCX or TXT files found in {data_folder}")
        return documents

    print(f"\n📂 Loading {len(files)} document(s) from {data_folder}...")

    for filename in files:
        file_path = os.path.join(data_folder, filename)
        text = load_document(file_path)
        if text:
            documents[filename] = text

    print(f"✅ Successfully loaded {len(documents)} document(s)\n")
    return documents