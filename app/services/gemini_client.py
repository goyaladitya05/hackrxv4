# === app/services/gemini_client.py ===
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable
import os

@traceable(name="GeminiClient")
def invoke_gemini(prompt: str, model_name: str = "gemini-2.5-pro") -> str:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm: BaseChatModel = ChatGoogleGenerativeAI(model=model_name)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
