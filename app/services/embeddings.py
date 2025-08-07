from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-base-en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        inputs = None
        outputs = None
        embeddings = None
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()
        finally:
            if torch.cuda.is_available():
                del inputs, outputs, embeddings  # explicitly free GPU tensors
                gc.collect()  # clean up CPU memory
                torch.cuda.empty_cache()  # release cached GPU memory

