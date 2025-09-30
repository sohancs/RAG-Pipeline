from dotenv import load_dotenv
import os


load_dotenv('rag_config.env')

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
HUG_FC_EMBEDDING_MODEL = os.getenv("HUG_FC_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLAMA_LLM_MODEL = os.getenv("LLAMA_LLM_MODEL", "llama3.2:1b")
PERSIST_PATH = os.getenv("PERSIST_PATH", "db/")
DIR_PATH = os.getenv("DIR_PATH", "data/")   
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_DOCKER_URL = os.getenv("OLLAMA_API_DOCKER_URL", "http://host.docker.internal:11434")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")