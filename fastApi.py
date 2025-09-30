from fastapi import FastAPI
from query import submit_query
from pydantic import BaseModel
from persistence import persist_enbeddings
from config import PERSIST_PATH, DIR_PATH

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

@api.get("/generate_embeddings/")
def generate_embeddings_api(overwrite: bool = False):
    """API endpoint to generate embeddings."""
    print(f"Generating & persisting embeddings... input overwrite flag : {overwrite}")
    persist_enbeddings(DIR_PATH, PERSIST_PATH, overwrite)
    return {
       "status": "Embeddings generated and persisted successfully"
    }


@api.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}