# === app/services/splitter.py ===
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)
