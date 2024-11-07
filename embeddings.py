#convert to data to chunks
 
from sentence_transformers import SentenceTransformer

def get_embeddings(chunks): 
    # Initialize Hugging Face embeddings with a model name  
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    
    # Get embeddings for a text query 
    embedding_vector =model.encode(chunks) 
    
    
    # Get embeddings for a document query 
    embedding_vector_docs =model.encode(chunks) 
    
    
    #Storing the chunks and their embeddings in a text file
    with open("./results/embeddings.txt", "w", encoding="utf-8")as f:
        f.write(f"Length of Embeddings {len(embedding_vector)}\n\n")
        for i in embedding_vector:
            f.write(f"{i}\n\n")
            
             
    '''----------------------------------------------------------'''
   
      
    return embedding_vector    
  