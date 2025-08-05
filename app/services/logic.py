# === app/services/logic.py ===
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import FAISSVectorStore
from app.services.gemini_client import invoke_gemini

embedding_model = EmbeddingService()


def retrieve_and_respond(text_chunks: list[str], questions: list[str]) -> dict:
    chunk_vectors = embedding_model.embed(text_chunks)
    vs = FAISSVectorStore(dim=chunk_vectors.shape[1])
    vs.add(chunk_vectors, text_chunks)

    results = []
    for q in questions:
        q_vec = embedding_model.embed([q])
        top_chunks = vs.search(q_vec, top_k=5)
        context = "\n".join([c for c, _ in top_chunks])
        prompt = f"You are a helpful assistant. Given the context below, answer the user's question in a single concise sentence. You may use commas but do not use bullet points or new lines.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
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
