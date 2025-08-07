from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from concurrent.futures import ProcessPoolExecutor
import os

def _invoke(prompt: str) -> str:
    try:
        llm = ChatOllama(model="mistral", temperature=0.2)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def batch_process_questions(prompts: list[str]) -> dict:
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        answers = list(executor.map(_invoke, prompts))
    return {"answers": answers}
