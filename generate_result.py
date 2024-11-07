from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma 
from langchain.schema import HumanMessage
import os
import shutil

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Load environment variables from .env file
load_dotenv()

# Constants
CHROMA_PATH = 'chroma'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("Google API key not found. Please set the GEMINI_API_KEY environment variable.")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
 
def generate_result(query_text, text):
    texts= text.split("\n")
    # Initializing the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    
        chunk_overlap=100,
        length_function=len,
        add_start_index=True 
    ) 
    
    # Initialize embedding function and Chroma vector store
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Split the text into chunks
    for text in texts:
        chunks = text_splitter.split_text(text)
        
        # Clear out DB if it exists
        if os.path.exists(CHROMA_PATH):
            vector_store = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)
        else:
            vector_store = Chroma.from_texts(chunks, embedding_function, persist_directory=CHROMA_PATH)
        
    
    print(f"Successfully stored {len(chunks)} documents and embeddings in ChromaDB!")
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)  # type: ignore
    
    try: 
        # Check if vector store is empty
        if vector_store is None:
            print("Vector store is empty.")
            return ""
        
        # Perform similarity search
        results = vector_store.similarity_search_with_relevance_scores(query_text, k=5)
        
        if len(results) == 0:
            print("No matching results found.")
            return ""
        
        print("Search Results:")
         
        # Prepare context text from results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt using context and question
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Create a HumanMessage instance for the prompt message
        prompt_message = [HumanMessage(content=prompt)]
        
        # Invoke the model
        answer = model.invoke(prompt_message)
        print("Answer:", answer.content)
        
        return answer.content  # Return the response content
    
    except Exception as e:
        print("Exception in loading and finding similarity:", e)
        return ""

# Example usage (uncomment to use)
# if __name__ == "__main__":
#     query = "Subjects in I year"
#     text = load_text()  # Make sure this is defined, or replace with your document-loading logic
#     generate_result(query, text)  



# Function to process the uploaded PDF file and extract text
