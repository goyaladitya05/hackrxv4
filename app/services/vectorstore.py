import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors: np.ndarray, texts: list[str]):
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_vector, top_k)
        return [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.texts)]

