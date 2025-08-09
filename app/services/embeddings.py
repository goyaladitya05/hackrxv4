import os
import torch
import numpy as np
import openai
import google.generativeai as genai
from itertools import cycle
from asyncio import Lock
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


class EmbeddingService:
    def __init__(self):
        self.local_model_name = "BAAI/bge-large-en-v1.5"
        self.openai_model_name = "text-embedding-3-large"
        self.gemini_model_name = "models/embedding-001"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === Flags to explicitly enable/disable embedders ===
        self.enable_local = True
        self.enable_openai = True
        self.enable_gemini = False

        # These track actual availability after loading
        self.use_local = False
        self.use_openai = False
        self.use_gemini = False

        self.executor = ThreadPoolExecutor(max_workers=4)

        # === Load Local Model ===
        if self.enable_local:
            try:
                self.model = SentenceTransformer(self.local_model_name, device=self.device)
                self.use_local = True
                print(f"[EmbeddingService] Loaded local model: {self.local_model_name} on {self.device}")
            except Exception as e:
                print(f"[EmbeddingService] Failed to load local model: {e}")

        # === Load OpenAI Keys ===
        if self.enable_openai:
            openai_keys = [
                os.getenv(f"OPENAI_API_KEY_{i}") for i in range(1, 11)
                if os.getenv(f"OPENAI_API_KEY_{i}")
            ]
            if openai_keys:
                self.openai_keys_cycle = cycle(openai_keys)
                self.openai_lock = Lock()
                self.openai_keys_list = openai_keys
                self.use_openai = True
                print(f"[EmbeddingService] Loaded {len(openai_keys)} OpenAI keys.")

        # === Load Gemini Keys ===
        if self.enable_gemini:
            gemini_keys = [
                os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 11)
                if os.getenv(f"GEMINI_API_KEY_{i}")
            ]
            if gemini_keys:
                self.gemini_keys_cycle = cycle(gemini_keys)
                self.gemini_lock = Lock()
                self.gemini_keys_list = gemini_keys
                self.use_gemini = True
                print(f"[EmbeddingService] Loaded {len(gemini_keys)} Gemini keys.")

    async def _get_next_openai_key(self):
        async with self.openai_lock:
            return next(self.openai_keys_cycle)

    async def _get_next_gemini_key(self):
        async with self.gemini_lock:
            return next(self.gemini_keys_cycle)

    async def _try_openai_embedding(self, texts):
        for _ in range(len(self.openai_keys_list)):
            key = await self._get_next_openai_key()
            try:
                openai.api_key = key
                resp = await openai.Embedding.acreate(model=self.openai_model_name, input=texts)
                print(f"[EmbeddingService] OpenAI key {key[:8]}... succeeded.")
                return np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
            except Exception as e:
                print(f"[EmbeddingService] OpenAI key {key[:8]}... failed: {e}")
        return None

    async def _try_gemini_embedding(self, texts):
        for _ in range(len(self.gemini_keys_list)):
            key = await self._get_next_gemini_key()
            try:
                genai.configure(api_key=key)
                resp = await self._run_in_executor(
                    genai.embed_content,
                    model=self.gemini_model_name,
                    content=texts
                )
                if "embedding" in resp:
                    embeddings = [resp["embedding"]["values"]]
                elif "embeddings" in resp:
                    embeddings = [e["values"] for e in resp["embeddings"]]
                else:
                    raise RuntimeError("Unexpected Gemini response format.")
                print(f"[EmbeddingService] Gemini key {key[:8]}... succeeded.")
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[EmbeddingService] Gemini key {key[:8]}... failed: {e}")
        return None

    async def _run_in_executor(self, func, *args, **kwargs):
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        # Try local first
        if self.use_local:
            try:
                return await self._run_in_executor(
                    self.model.encode,
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            except torch.cuda.OutOfMemoryError:
                print("[EmbeddingService] GPU OOM — disabling local and retrying.")
                torch.cuda.empty_cache()
                self.use_local = False
            except Exception as e:
                print(f"[EmbeddingService] Local embedding error: {e}")
                self.use_local = False

        # Try OpenAI
        if self.use_openai:
            result = await self._try_openai_embedding(texts)
            if result is not None:
                return result
            else:
                print("[EmbeddingService] All OpenAI keys failed or unavailable.")

        # Try Gemini
        if self.use_gemini:
            result = await self._try_gemini_embedding(texts)
            if result is not None:
                return result
            else:
                print("[EmbeddingService] All Gemini keys failed.")

        raise RuntimeError("No embedding source available (local, OpenAI, or Gemini).")
