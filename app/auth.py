# === app/auth.py ===
import os
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

load_dotenv()
EXPECTED_TOKEN = os.getenv("BEARER_TOKEN")

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
    return token
