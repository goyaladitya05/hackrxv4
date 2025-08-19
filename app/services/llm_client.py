import os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable
from openai import RateLimitError
import time
import google.generativeai as genai

load_dotenv()
openai_clients = {}

@traceable(name="Testing-OpenAIClient")
def invoke_openai(prompt: str, api_key: str, model_name: str = "gpt-4o", stream: bool = False) -> str:
    client = get_openai_client(api_key)
    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True,
        )
        collected = [chunk.choices[0].delta.content or "" for chunk in response]
        return "".join(collected).strip()
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

def get_openai_client(api_key: str) -> OpenAI:
    if api_key not in openai_clients:
        openai_clients[api_key] = OpenAI(api_key=api_key)
    return openai_clients[api_key]

@traceable(name="Testing-GeminiClient")
@traceable(name="Testing-GeminiClient")
def invoke_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.5-pro") -> str:
    genai.configure(api_key=api_key)
    chat = genai.GenerativeModel(model_name).start_chat()
    response = chat.send_message(prompt)

    # Check if candidates exist and contain text parts before accessing .text
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        # candidate may or may not have a .text attribute, check safely
        if hasattr(candidate, "text") and candidate.text:
            return candidate.text.strip()
    # Fallback return if no valid text is found
    return "There's no answer to this in the provided context"


# === Worker Logic ===
def process_with_key(args: Tuple[int, str, str, bool]) -> Dict:
    index, prompt, model_name, openai_enabled = args
    openai_key = os.getenv(f'OPENAI_API_KEY_{index + 1}')
    gemini_key = os.getenv(f'GEMINI_API_KEY_{index + 1}')

    # Try OpenAI
    if openai_enabled and openai_key:
        try:
            answer = invoke_openai(prompt, openai_key, model_name)
            return {'question': prompt, 'answer': answer, 'status': 'openai-gpt4-success'}
        except RateLimitError:
            print(f"[RateLimit] GPT-4 rate limit for key {index + 1}, falling back to GPT-3.5.")
            try:
                answer = invoke_openai(prompt, openai_key, model_name="gpt-3.5-turbo")
                return {'question': prompt, 'answer': answer, 'status': 'openai-gpt35-fallback'}
            except Exception as e2:
                print(f"[Fallback] GPT-3.5 also failed: {e2}")
        except Exception as e:
            print(f"[OpenAI Error] GPT-4o failed for key {index + 1}: {e}")

    # Try Gemini 2.5 Pro
    if gemini_key:
        try:
            answer = invoke_gemini(prompt, gemini_key, model_name="gemini-2.5-pro")
            return {'question': prompt, 'answer': answer, 'status': 'gemini-pro-success'}
        except Exception as e:
            print(f"[Gemini-Pro Error] Key {index + 1} failed: {e}")

            # Final fallback: Gemini 2.5 Flash
            try:
                answer = invoke_gemini(prompt, gemini_key, model_name="gemini-2.5-flash")
                return {'question': prompt, 'answer': answer, 'status': 'gemini-flash-fallback'}
            except Exception as e2:
                print(f"[Gemini-Flash Error] Key {index + 1} failed: {e2}")

    return {
        'question': prompt,
        'answer': "All model invocations failed.",
        'status': 'error'
    }

# === Batch Processor ===
def batch_process_questions(questions: List[str], model_name: str = "gpt-4o") -> List[Dict]:
    openai_keys = [os.getenv(f'OPENAI_API_KEY_{i}') for i in range(1, 11)]
    gemini_keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 11)]

    openai_enabled = any(openai_keys)
    gemini_enabled = any(gemini_keys)

    if not openai_enabled and not gemini_enabled:
        print("[Info] No OpenAI or Gemini keys found. All requests will fail.")

    num_slots = sum([openai_enabled, gemini_enabled]) or 1

    tasks = [
        (i % num_slots, question, model_name, openai_enabled)
        for i, question in enumerate(questions)
    ]

    with ThreadPoolExecutor(max_workers=min(num_slots, len(questions))) as executor:
        results = list(executor.map(process_with_key, tasks))

    return results
