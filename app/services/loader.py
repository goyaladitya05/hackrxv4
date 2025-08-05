# === app/services/loader.py ===
import os
import tempfile
import requests
#import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from email import message_from_string

def download_file(url: str) -> str:
    url = str(url)  # Convert HttpUrl to plain string
    suffix = url.split("?")[0].split(".")[-1]
    response = requests.get(url)
    response.raise_for_status()
    suffix = url.split("?")[0].split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(response.content)
        return tmp.name

import pdfplumber # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(filter(None, text))  # removes None pages

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_email(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        msg = message_from_string(f.read())
        body = msg.get_payload(decode=True)
        soup = BeautifulSoup(body or "", "html.parser")
        return soup.get_text()

def load_from_url(url: str) -> str:
    path = download_file(url)
    if path.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        return extract_text_from_docx(path)
    elif path.endswith(".eml"):
        return extract_text_from_email(path)
    else:
        raise ValueError("Unsupported file format")

