import numpy as np
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import FAISSVectorStore
from app.services.gemini_client import invoke_gemini

embedding_model = EmbeddingService()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_and_respond(text_chunks: list[str], questions: list[str]) -> dict:
    chunk_vectors = embedding_model.embed(text_chunks)
    results = []
    for q in questions:
        q_vec = embedding_model.embed([q])[0]
        # Compute cosine similarity with each chunk
        similarities = [cosine_similarity(q_vec, chunk_vec) for chunk_vec in chunk_vectors]
        # Get top 5 most similar chunks
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_chunks = [text_chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)
        prompt = fprompt = (
    "You are a helpful assistant. ONLY use the information provided in the context below to answer the user's question. "
    "Do not use any outside knowledge. If the answer is not in the context, reply with 'I don't know.' "
    "Respond in a single sentence, or two sentences maximum. Be clear, concise, and factual. "
    "Avoid bullet points, line breaks, and unnecessary details. Use commas if needed to keep the sentence compact.\n\n"
    f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
)
        response = invoke_gemini(prompt)
        results.append({"question": q, "answer": response})

    return restructure_response(results)

def restructure_response(response) -> dict:
    if isinstance(response, dict):
        response = response.get("answers", [])

    answers = [
        item["answer"].strip()
        for item in response
        if item.get("answer", "").strip()
    ]
    return {"answers": answers}