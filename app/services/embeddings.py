from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-base-en"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.gpu_available = torch.cuda.is_available()
        self.model_gpu = None
        self.model_cpu = None

        if self.gpu_available:
            try:
                self.model_gpu = AutoModel.from_pretrained(model_name).to("cuda")
                self.model_gpu.eval()
                print(f"[Init] Loaded model on GPU: {model_name}")
            except Exception as e:
                print(f"[Warning] Failed to load model on GPU: {e}")
                self.gpu_available = False

        # Always load CPU model for fallback
        self.model_cpu = AutoModel.from_pretrained(model_name).to("cpu")
        self.model_cpu.eval()
        print(f"[Init] Loaded model on CPU: {model_name}")

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            device = "cuda" if self.gpu_available else "cpu"
            model = self.model_gpu if self.gpu_available else self.model_cpu
            print(f"[Embed] Using {device.upper()} for embedding.")
            return self._embed(texts, device=device, model=model)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[Fallback] CUDA OOM – retrying on CPU...")
                return self._embed(texts, device="cpu", model=self.model_cpu)
            else:
                raise e

    def _embed(self, texts: list[str], device: str, model: AutoModel) -> np.ndarray:
        inputs = None
        outputs = None
        embeddings = None
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()
        finally:
            if device == "cuda":
                print("[Cleanup] Clearing CUDA memory...")
                del inputs, outputs, embeddings
                gc.collect()
                torch.cuda.empty_cache()
