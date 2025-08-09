from typing import Dict, List
import numpy as np
import time
from app.services.embeddings import EmbeddingService
from app.services.llm_client import batch_process_questions

embedding_model = EmbeddingService()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def retrieve_and_respond(text_chunks: List[str], questions: List[str]) -> Dict[str, List[str]]:
    start = time.time()

    # Await embedding calls because embed_texts() is async
    chunk_vectors = await embedding_model.embed_texts(text_chunks)
    question_vectors = await embedding_model.embed_texts(questions)

    print(f"[Embedding] Took {time.time() - start:.2f}s")

    prompts = []
    for q, q_vec in zip(questions, question_vectors):
        similarities = [cosine_similarity(q_vec, chunk_vec) for chunk_vec in chunk_vectors]
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_chunks = [text_chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)

        prompt = build_concise_prompt(context, q)
        prompts.append(prompt)

    # If batch_process_questions is async, await it
    results = await batch_process_questions(prompts) if callable(getattr(batch_process_questions, "__await__", None)) else batch_process_questions(prompts)

    return restructure_response(results)

def build_concise_prompt(context: str, question: str) -> str:
    return (
        "You are an insurance policy expert. Answer the question using ONLY the information from the context below. "
        "The context may include tables, bullet points, or structured formats — carefully analyze and extract relevant details from these. "
        "Keep your response to 1 sentence only, unless essential context (e.g. legal names or conditions) requires 2, that includes essential data like numbers, time periods, limits, or conditions. "
        "Only provide information that is directly stated or can be clearly inferred from the context. "
        "Do not include assumptions or external knowledge. "
        "If the answer is truly not present in the context — including in tables or structured formats — reply with exactly: "
        "'There's no answer to this in the provided context'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def restructure_response(results: List[Dict]) -> Dict[str, List[str]]:
    return {
        "answers": [result["answer"] for result in results]
    }
