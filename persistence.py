from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_directory
from langchain_chroma import Chroma
from splitter import split_document
from config import PERSIST_PATH, EMBEDDING_MODEL, DIR_PATH
import time

def get_embedding_model():
    """Get embedding model."""
    model_name = EMBEDDING_MODEL
    return HuggingFaceEmbeddings(model_name=model_name)


def persist_enbeddings(dir_path: str, persist_path: str) :
    """persist embeddings to disk. """

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

    print (f"TIme taken to create embeddings & persisting : {end_time - start_time : .2f} seconds.")

    #vectorstore.persist() #not required in new version of ChromaDB
    print(f"Data has been pesisted successfully to filepath {persist_path}")


    return vectorstore

if __name__ == "__main__":
    persist_enbeddings(dir_path=DIR_PATH, persist_path=PERSIST_PATH) #start the process