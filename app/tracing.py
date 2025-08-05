# === app/tracing.py ===
import os
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled
from app.config import settings

os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

client = Client()

# Usage: with tracing_v2_enabled():
#     chain.invoke({...})
