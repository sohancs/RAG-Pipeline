#from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_directory
from langchain_community.vectorstores import Chroma
from splitter import split_document
from config import PERSIST_PATH, HUG_FC_EMBEDDING_MODEL, DIR_PATH, OPENAI_API_KEY, EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings
import time
import shutil
import os

def get_embedding_model():
    """Get embedding model."""
    print(f"Current Embedding provider : {EMBEDDING_PROVIDER}")
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key= OPENAI_API_KEY)
    # else:
    #     return HuggingFaceEmbeddings(model_name=HUG_FC_EMBEDDING_MODEL)


def persist_enbeddings(dir_path: str, persist_path: str, overwrite_flag: bool = True) :
    """persist embeddings to disk. """

    if not overwrite_flag and os.path.exists(persist_path) :
        #remove existing directory
        shutil.rmtree(persist_path)
        print (f"Existing directory {persist_path} removed.")

    documents = load_directory(dir_path)

    if not documents:
        print("No documents to persist")
        raise ValueError("No documents to persist")
    
    chunks = split_document(documents)

    embeddings = get_embedding_model()

    start_time = time.perf_counter()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    end_time = time.perf_counter()

    print (f"Time taken to create embeddings & persisting : {end_time - start_time : .2f} seconds.")

    #vectorstore.persist() #not required in new version of ChromaDB
    print(f"Data has been pesisted successfully to filepath {persist_path}")


    return vectorstore

if __name__ == "__main__":
    persist_enbeddings(dir_path=DIR_PATH, persist_path=PERSIST_PATH, overwrite_flag=False) #start the process