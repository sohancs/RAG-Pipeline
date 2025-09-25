from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

textSplitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def split_document(document) : 
    """Split documents into small chunks."""
    chunks = textSplitter.split_documents(document)
    print (f"chunk size {len(chunks)}")
    return chunks


