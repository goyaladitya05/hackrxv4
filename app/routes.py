from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from app.model import QueryRequest, QueryResponse
from app.auth import verify_token
from app.services.loader import load_from_url
from app.services.splitter import split_text
from app.services.logic import retrieve_and_respond

router = APIRouter()

@router.post("/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest, authorization: str = Header(...)):
    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    full_text = load_from_url(request.documents)
    chunks = split_text(full_text)
    answers = retrieve_and_respond(chunks, request.questions)
    return answers  # optionally wrap in JSONResponse(content=answers)

@router.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(content={"status": "ok", "message": "Server is healthy"})

@router.get("/hackrx/run")
async def wrong_method():
    return JSONResponse(content={"status": "ok", "message": "Send a POST request. GET not allowed on this route"})
