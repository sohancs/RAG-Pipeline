from persistence import get_embedding_model
#from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from config import PERSIST_PATH, LLM_MODEL, OLLAMA_API_URL, OLLAMA_API_DOCKER_URL
import time
import argparse
from ollama_wrapper import OllamaWrapper

def get_vectorstore(persist_path: str) :

    """Get vectorstore."""
    embeddings = get_embedding_model()
    vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectorstore

def query_model(persist_path: str) :

    """Query model & return reponse."""

    #calling ollama build in function
    #llm_model = ChatOllama(model=LLM_MODEL, temperature=0)

    #calling custom wrapper to use ollama api through docker container
    llm_model = OllamaWrapper(model_name=LLM_MODEL, base_url=OLLAMA_API_DOCKER_URL)

    prompt = PromptTemplate(
        template=(
                "You are an assistant that answers user questions using the provided context.\n"
                "If the answer is not present in the context, say 'I don't know'.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer :"
        ), 
        input_variables=["context", "question"]
    )

    retriever = get_vectorstore(persist_path=persist_path).as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa;

def submit_query(query: str) :
    """Submit query to model."""
    persist_path = PERSIST_PATH

    qa = query_model(persist_path=persist_path)

    try:
        start_time = time.perf_counter()
       
        response = qa.invoke({"query": query})

        end_time = time.perf_counter()

        print (f"Time taken to create generate ans : {end_time - start_time : .2f} seconds.")
        print("Response: ", response['result'])
       
        #print("Source documents: ", response['source_documents'])
        return response
    except Exception as e:
        print("Error during query:", e)
        
   

#create argparser
def argparser():
    """Create argparser."""
    parser = argparse.ArgumentParser(description="Ask question here")
    parser.add_argument("query", type=str, help="Query to ask the model")
    
    #parse args
    args = parser.parse_args()
    query = args.query

    return query




if __name__ == "__main__" :
    query = argparser()
    submit_query(query) #start the process