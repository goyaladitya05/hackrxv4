from typing import Dict, List
import numpy as np
from app.services.embeddings import EmbeddingService
from app.services.llm_client import batch_process_questions

embedding_model = EmbeddingService()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_and_respond(text_chunks: List[str], questions: List[str]) -> Dict[str, List[str]]:
    chunk_vectors = embedding_model.embed(text_chunks)
    prompts = []

    for q in questions:
        q_vec = embedding_model.embed([q])[0]
        similarities = [cosine_similarity(q_vec, chunk_vec) for chunk_vec in chunk_vectors]
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_chunks = [text_chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)

        prompt = build_concise_prompt(context, q)
        prompts.append(prompt)

    results = batch_process_questions(prompts)
    return restructure_response(results)


def build_concise_prompt(context: str, question: str) -> str:
    return (
        "You are an insurance policy expert. Answer the question using ONLY the information from the context below. "
        "Provide a clear, concise answer that includes essential details like numbers, time periods, and key conditions. "
        "Keep your response to 1-2 sentences maximum. Focus on the most important information. "
        "Do not include unnecessary details, explanations, or conditions unless they are core to the answer. "
        "If the answer is not in the context, reply with 'There's no answer to this in the provided context'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_prompt(context: str, question: str) -> str:
    return build_concise_prompt(context, question)


def process_questions(context: str, questions: List[str]) -> Dict[str, List[str]]:
    prompts = []

    for q in questions:
        prompt = (
            "You are an insurance policy expert. Answer the question using ONLY the information from the context below. "
            "Provide a clear, concise answer that includes essential details like numbers, time periods, and key conditions. "
            "Keep your response to 1-2 sentences maximum. Focus on the most important information. "
            "Do not include unnecessary details, explanations, or extensive conditions unless they are core to the answer. "
            "If the answer is not in the context, reply with 'There's no answer to this in the provided context'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            "Answer:"
        )
        prompts.append(prompt)

    results = batch_process_questions(prompts)
    return {
        "answers": [result["answer"] for result in results]
    }


def process_questions_with_examples(context: str, questions: List[str]) -> Dict[str, List[str]]:
    """
    Version with specific examples to guide the response format
    """
    example_answers = """
Example of good concise answers:
Q: What is the grace period for premium payment?
A: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Q: What is the waiting period for pre-existing diseases?  
A: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

Q: Does the policy cover maternity expenses?
A: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.
"""

    prompts = []

    for q in questions:
        prompt = (
            "You are an insurance policy expert. Answer questions concisely using ONLY the context provided. "
            "Follow the style of these examples - include key details but keep responses brief and focused:\n\n"
            f"{example_answers}\n\n"
            "Now answer the following question in the same concise style:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            "Answer:"
        )
        prompts.append(prompt)

    results = batch_process_questions(prompts)
    return {
        "answers": [result["answer"] for result in results]
    }


def restructure_response(results: List[Dict]) -> Dict[str, List[str]]:
    return {
        "answers": [result["answer"] for result in results]
    }