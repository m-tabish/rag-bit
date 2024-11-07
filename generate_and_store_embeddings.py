#Generating embeddings and storing all the chunks in DB
from sentence_transformers import SentenceTransformer 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil

CHROMA_PATH = 'chroma'

def generate_and_store_embeddings(documents):  
    
    
    #Clearing out DB 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Wrap each chunk in a Document object
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in documents]
    embeddings= model.encode(texts)
    print("embeds")
    print(embeddings)
    
    # Create the Chroma vector store using the embeddings
    vector_store = Chroma.from_documents(documents, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), persist_directory=CHROMA_PATH)
    
    print(f"Successfully stored {len(documents)} documents and embeddings in ChromaDB!")
    return vector_store

     