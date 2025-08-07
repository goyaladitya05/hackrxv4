import os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import openai
import google.generativeai as genai
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

# === OpenAI PRIMARY ===
@traceable(name="OpenAIClient")
def invoke_openai(prompt: str, api_key: str, model_name: str = "gpt-4") -> str:
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


# === Gemini Fallback ===
@traceable(name="GeminiClient")
def invoke_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=api_key)
    chat = genai.GenerativeModel(model_name).start_chat()
    response = chat.send_message(prompt)
    return response.text.strip()


# === Worker Logic ===
def process_with_key(args: Tuple[int, str, str, bool]) -> Dict:
    index, prompt, model_name, openai_enabled = args

    openai_key = os.getenv(f'OPENAI_API_KEY_{index + 1}')
    gemini_key = os.getenv(f'GEMINI_API_KEY_{index + 1}')

    if openai_enabled and openai_key:
        try:
            answer = invoke_openai(prompt, openai_key, model_name)
            return {
                'question': prompt,
                'answer': answer,
                'status': 'openai-success'
            }
        except Exception as e:
            print(f"[Fallback] OpenAI Key {index + 1} failed: {e}")

    if gemini_key:
        try:
            answer = invoke_gemini(prompt, gemini_key)
            return {
                'question': prompt,
                'answer': answer,
                'status': 'gemini-fallback' if openai_enabled else 'gemini-success'
            }
        except Exception as e2:
            return {
                'question': prompt,
                'answer': f"Gemini failed: {e2}",
                'status': 'error'
            }

    return {
        'question': prompt,
        'answer': f"No available API key at index {index + 1}",
        'status': 'error'
    }


# === Batch Processor with OpenAI+Gemini or Gemini-only mode ===
def batch_process_questions(questions: List[str], model_name: str = "gpt-3.5-turbo") -> List[Dict]:
    openai_keys = [os.getenv(f'OPENAI_API_KEY_{i}') for i in range(1, 11)]
    gemini_keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 11)]

    openai_enabled = any(openai_keys)
    gemini_enabled = any(gemini_keys)

    if not openai_enabled and not gemini_enabled:
        raise ValueError("No OpenAI or Gemini keys found in environment variables")

    num_slots = max(len([k for k in openai_keys if k]), len([k for k in gemini_keys if k]))

    tasks = [
        (i % num_slots, question, model_name, openai_enabled)
        for i, question in enumerate(questions)
    ]

    with ThreadPoolExecutor(max_workers=min(num_slots, len(questions))) as executor:
        results = list(executor.map(process_with_key, tasks))

    return results