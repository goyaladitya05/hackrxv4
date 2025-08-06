from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    #clause: Optional[str] = None
    #explanation: Optional[str] = None

class QueryResponse(BaseModel):
    answers: List[str]