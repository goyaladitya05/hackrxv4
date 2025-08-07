from app.services.vectorstore import FAISSVectorStore
from app.services.embeddings import EmbeddingService
from app.services.llm_client import batch_process_questions

embedding_model = EmbeddingService()

def build_prompt(context: str, question: str) -> str:
    return (
        "You are an expert assistant. ONLY use the given context to answer. "
        "Be concise, factual, and avoid unnecessary details. Answer in 1-2 clear sentences. "
        "If no relevant info is found, respond: 'There's no answer to this in the provided context'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

def retrieve_and_respond(chunks: list[str], questions: list[str], top_k: int = 5):
    chunk_vectors = embedding_model.embed(chunks)
    vector_store = FAISSVectorStore(dim=chunk_vectors.shape[1])
    vector_store.add(chunk_vectors, chunks)

    prompts = []
    for q in questions:
        q_vector = embedding_model.embed([q])[0].reshape(1, -1)
        top_chunks = vector_store.search(q_vector, top_k=top_k)
        context = "\n".join([chunk for chunk, _ in top_chunks])
        prompt = build_prompt(context, q)
        prompts.append(prompt)

    return batch_process_questions(prompts)