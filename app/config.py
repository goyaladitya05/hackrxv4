# === app/config.py ===
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BEARER_TOKEN: str = os.getenv("BEARER_TOKEN")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "hackrx-query-retrieval")

settings = Settings()
