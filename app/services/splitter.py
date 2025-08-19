from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(
    text: str,
    chunk_size: int = 1600,
    chunk_overlap: int = 300
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", ",", " "]
    )
    return splitter.split_text(text)