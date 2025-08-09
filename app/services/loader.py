# === app/services/loader.py ===
import os
import tempfile
import requests
import mimetypes
import docx
import csv
import pandas as pd
from bs4 import BeautifulSoup
from email import policy
from email import message_from_binary_file
import pdfplumber
from PIL import Image
import pytesseract

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx", ".eml",
    ".jpg", ".jpeg", ".png", ".tiff"
}

# ---------- File Download ----------
def download_file(url: str) -> str:
    """
    Download file from a regular URL or Google Drive link.
    Returns local file path.
    """
    url = str(url)

    # Handle Google Drive links
    if "drive.google.com" in url:
        file_id = None
        if "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]

        if file_id:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()

    # Guess extension
    ext = guess_extension(url, response)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(response.content)
        return tmp.name

def guess_extension(url: str, response) -> str:
    """Guess file extension from URL or Content-Type header."""
    if "." in url.split("?")[0]:
        ext = "." + url.split("?")[0].split(".")[-1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return ext

    ctype = response.headers.get("Content-Type")
    if ctype:
        ext = mimetypes.guess_extension(ctype.split(";")[0].strip())
        if ext:
            return ext

    return ".tmp"

# ---------- Extractors ----------
def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
            else:
                # OCR fallback for scanned pages
                img = page.to_image(resolution=300).original
                pil_img = Image.fromarray(img)
                ocr_text = pytesseract.image_to_string(pil_img)
                if ocr_text.strip():
                    text.append(ocr_text)
    return "\n".join(text)

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_csv(file_path: str) -> str:
    with open(file_path, newline="", encoding="utf-8", errors="ignore") as csvfile:
        return "\n".join([" | ".join(row) for row in csv.reader(csvfile)])

def extract_text_from_xlsx(file_path: str) -> str:
    dfs = pd.read_excel(file_path, sheet_name=None)
    return "\n".join(df.to_csv(index=False) for df in dfs.values())

def extract_text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

def extract_text_from_email(file_path: str) -> str:
    texts = []
    with open(file_path, "rb") as f:
        msg = message_from_binary_file(f, policy=policy.default)

    for part in msg.walk():
        content_disposition = part.get_content_disposition()
        content_type = part.get_content_type()

        if content_disposition == "attachment":
            filename = part.get_filename()
            if filename:
                suffix = os.path.splitext(filename)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(part.get_payload(decode=True))
                    texts.append(load_from_path(tmp.name))
        elif content_type in ["text/plain", "text/html"]:
            payload = part.get_payload(decode=True)
            if payload:
                if content_type == "text/html":
                    soup = BeautifulSoup(payload, "html.parser")
                    texts.append(soup.get_text())
                else:
                    texts.append(payload.decode(errors="ignore"))

    return "\n".join(filter(None, texts))

# ---------- Main Loader ----------
def load_from_path(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".csv":
        return extract_text_from_csv(path)
    elif ext == ".xlsx":
        return extract_text_from_xlsx(path)
    elif ext in {".jpg", ".jpeg", ".png", ".tiff"}:
        return extract_text_from_image(path)
    elif ext == ".eml":
        return extract_text_from_email(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_from_url(url: str) -> str:
    path = download_file(url)
    return load_from_path(path)
