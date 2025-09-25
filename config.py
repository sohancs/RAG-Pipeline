from dotenv import load_dotenv
import os


load_dotenv('rag_config.env')

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
PERSIST_PATH = os.getenv("PERSIST_PATH", "db/")
DIR_PATH = os.getenv("DIR_PATH", "data/")                           