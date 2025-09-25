from persistence import get_embedding_model
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from config import PERSIST_PATH, LLM_MODEL
import time
import argparse

def get_vectorstore(persist_path: str) :

    """Get vectorstore."""
    embeddings = get_embedding_model()
    vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectorstore

def query_model(persist_path: str) :

    """Query model & return reponse."""

    llm_model = ChatOllama(model=LLM_MODEL, temperature=0)

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

    start_time = time.perf_counter()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa;

def tester_function() :
    """Test function."""
    persist_path = PERSIST_PATH

    qa = query_model(persist_path=persist_path)

    #query = "How is Qualcomm company to work for?" #"Does company allow remote work?"  #"What is leave policy in Sellthru?"

    query = argparser()

    try:
        start_time = time.perf_counter()
       
        response = qa.invoke({"query": query})

        end_time = time.perf_counter()

        print (f"Time taken to create generate ans : {end_time - start_time : .2f} seconds.")
        print("Response: ", response['result'])
       
        #print("Source documents: ", response['source_documents'])
    except Exception as e:
        print("Error during query:", e)
        
   

#create argparser
def argparser():
    parser = argparse.ArgumentParser(description="Ask question here")
    parser.add_argument("query", type=str, help="Query to ask the model")
    
    #parse args
    args = parser.parse_args()
    query = args.query

    return query

if __name__ == "__main__" :
    tester_function() #start the test