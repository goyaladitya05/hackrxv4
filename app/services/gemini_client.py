import os
from typing import Dict, List, Tuple
from multiprocessing import Pool

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

@traceable(name="Geminiv4Client")
def invoke_gemini(prompt: str, api_key: str = None, model_name: str = "gemini-2.5-flash") -> str:
    # Use provided API key or fallback to environment variable
    key = api_key or os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm: BaseChatModel = ChatGoogleGenerativeAI(model=model_name,api_key=key)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def process_with_key(args: tuple) -> Dict:
    """Process a single question with given API key"""
    api_key, prompt, model_name = args
    try:
        response = invoke_gemini(prompt, api_key, model_name)
        return {
            'question': prompt,
            'answer': response,
            'status': 'success'
        }
    except Exception as e:
        return {
            'question': prompt,
            'answer': str(e),
            'status': 'error'
        }


def batch_process_questions(questions: List[str], model_name: str = "gemini-2.5-flash") -> List[Dict]:
    api_keys = [
        os.getenv(f'GEMINI_API_KEY_{i}')
        for i in range(1, 11)
        if os.getenv(f'GEMINI_API_KEY_{i}')
    ]

    if not api_keys:
        raise ValueError("No Gemini API keys found in environment variables")

    tasks = [
        (api_keys[i % len(api_keys)], question, model_name)
        for i, question in enumerate(questions)
    ]

    with ThreadPoolExecutor(max_workers=min(len(api_keys), len(questions))) as executor:
        results = list(executor.map(process_with_key, tasks))

    return results

def process_questions(context: str, questions: List[str]) -> Dict[str, List[str]]:
    prompts = [
        f"Avoid bullet points, line breaks, and unnecessary details. Use commas if needed to keep the sentence compact.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        for q in questions
    ]
    
    results = batch_process_questions(prompts)
    return {
        "answers": [result["answer"] for result in results]
    }