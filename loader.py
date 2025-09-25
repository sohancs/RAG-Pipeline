from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_file(file_path: str) -> Document :
    """Load a file and return its content as a Document."""

    loader = PyPDFLoader(file_path)
    documents = loader.load();

    return documents

def load_directory(file_path : str) :
    """Load all files in a directory and return their content as a list of Documents."""
    
    loader = DirectoryLoader(
        path=file_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
        )
    
    documents = loader.load()
    print ("documents size: ", len(documents))

    return documents



    
#print (load_file("data/**"))
#print (load_directory("data/"))
