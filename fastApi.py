from fastapi import FastAPI
from query import submit_query
from pydantic import BaseModel

api = FastAPI(title= "API for RAG Pipeline", version="1.0")

class QueryDTO(BaseModel):
    """Model for request payload to submit query API"""
    query : str


@api.post("/submit_query")
def submit_query_api(request: QueryDTO):
    """API endpoint to submit query."""
    response = submit_query(request.query)
    return {
        "query": request.query,
        "answer": response['result'],
        #"source_documents": response['source_documents']
    }


@api.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}