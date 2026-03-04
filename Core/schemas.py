from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    database_id: int
    message: str
