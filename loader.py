#from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
import os

def load_file(file_path: str) :
    """Load a file and return its content as a Document."""

    loader = PyPDFLoader(file_path)
    documents = loader.load();

    return documents

def load_directory(dir_path : str) :
    """Load all files in a directory and return their content as a list of Documents."""

    documents = []

    for file in os.listdir(dir_path) :
        file_name = os.path.join(dir_path, file)
        if(file.endswith(".pdf")) :
            loader = PyPDFLoader(file_name)
            docs = loader.load()
            documents.extend(docs)

        elif(file.endswith(".txt")) :
            loader = TextLoader(file_name)
            docs = loader.load()
            documents.extend(docs)
    
    print (f"No. of files loaded : {len(os.listdir(dir_path))}, No. of documents : {len(documents)}")

    return documents



    
#print (load_file("data/**"))
#print (load_directory("data/"))
