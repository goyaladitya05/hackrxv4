import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import asyncio

class EmbeddingService:
    def __init__(self):
        self.local_model_name = "all-roberta-large-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.executor = ThreadPoolExecutor(max_workers=1)  # single-threaded executor
        self.model = SentenceTransformer(self.local_model_name, device=self.device)
        print(f"[EmbeddingService] Loaded local model: {self.local_model_name} on {self.device}")

        self.embedding_queue = asyncio.Queue()
        self.embedding_task = asyncio.create_task(self._embedding_worker())

        self._last_doc_id = None
        self._last_doc_embeddings = None

    async def embed_document(self, doc_id: str, chunks: list[str]) -> np.ndarray:
        # Return cached if same document id requested consecutively
        if doc_id == self._last_doc_id and self._last_doc_embeddings is not None:
            print("[EmbeddingService] Using cached document embeddings.")
            return self._last_doc_embeddings

        fut = asyncio.get_running_loop().create_future()
        await self.embedding_queue.put((doc_id, chunks, fut))
        embeddings = await fut
        return embeddings

    async def _embedding_worker(self):
        while True:
            doc_id, chunks, fut = await self.embedding_queue.get()
            try:
                print(f"[EmbeddingWorker] Embedding document id: {doc_id}")
                embeddings = await self._run_in_executor(
                    self.model.encode,
                    chunks,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                # Cache embeddings for next request
                self._last_doc_id = doc_id
                self._last_doc_embeddings = embeddings

                fut.set_result(embeddings)
            except Exception as e:
                print(f"[EmbeddingWorker] Embedding failed: {e}")
                if not fut.done():
                    fut.set_exception(e)
            finally:
                self.embedding_queue.task_done()

    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))

    async def embed_questions(self, questions: list[str]) -> np.ndarray:
        # You can generate a dummy doc_id or a hash of questions to identify duplicates
        doc_id = "questions:" + str(hash(tuple(questions)))
        return await self.embed_document(doc_id, questions)