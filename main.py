import os
from dotenv import load_dotenv     
from create_chunks import create_chunks 
from generate_and_store_embeddings import generate_and_store_embeddings
from langchain_doc_loader import load_documents
 
from loading_db import load_db
# Load environment variables from .env file
load_dotenv() 
 
def main():
    
    # Loading the document
    # documents = load_documents() 
    # '''----------------------------------------------------------'''
    
    # # Splitting the document into chunks
    # kb_chunks = create_chunks(documents) 
    
    # # Save chunks to a file only if chunks exist
    # if kb_chunks:  
    #     print("Chunks written to ./results/chunks.txt")
    # else:
    #     print("No text chunks created.")
        
    # '''----------------------------------------------------------''' 
    # #Generate embeddings and store in ChromaDB
    
    # vector_store = generate_and_store_embeddings(documents)

    
    '''----------------------------------------------------------'''
    #Retreiving results (conducting similarity search)
    query_text = input("Enter your query\n")
    result = load_db(query_text) 
    print(result)
    # Retrieve the GEMINI_API_KEY from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        print("API Key:", api_key)
    else:
        print("API Key not found.")

if __name__ == "__main__":
    main()
