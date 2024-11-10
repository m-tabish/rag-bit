# Generate Result

import PyPDF2
from langchain.schema import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma 
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

# Function to process and extract text from a PDF file and return Document objects
# def extract_text_from_pdf_to_documents(pdf_path):
#     with open(pdf_path, "rb") as file:  
#         reader = PyPDF2.PdfReader(file)
#         documents = []
#         for page_num, page in enumerate(reader.pages):
#             text = page.extract_text()
#             if text:
#                 # Create a Document object for each page with the extracted text
#                 doc = Document(page_content=text)
#                 documents.append(doc)
#     return documents

# Function to generate results from the documents
def generate_result(query_text, documents): 
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
     
    vector_store = Chroma.from_documents(documents, embedding_function, persist_directory=CHROMA_PATH)

    print(f"Successfully stored {len(documents)} documents and embeddings in ChromaDB!")
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)  # type: ignore
    
    try: 
        # Check if vector store is empty
        if vector_store is None:
            print("Vector store is empty.")
            return ""
        
        # Perform similarity search
        results = vector_store.similarity_search_with_relevance_scores(query_text, k=10)
        
        if len(results) == 0:
            print("No matching results found.")
            return ""
        
        # print("Search Results:")
        
        # Print out each result (document and relevance score)
        # for idx, (doc, score) in enumerate(results, 1):
        #     print(f"Result {idx}:")
        #     print(f"Relevance Score: {score}")
        #     print(f"Document Content: {doc.page_content[:300]}...")  # Print first 300 chars of the document content
        #     print("="*50) 
        
        # Prepare context text from results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt using context and question
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Create a HumanMessage instance for the prompt message
        prompt_message = [HumanMessage(content=prompt)]
        
        # Invoke the model
        answer = model.invoke(prompt_message)
         
        return answer.content  # Return the response content
    
    except Exception as e:
        print("Exception in loading and finding similarity:", e)
        return ""

# Example of using the function
# if __name__ == "__main__":
#     # Sample query
#     query = "What is deep learning?"
    
#     # Load documents from the PDF
#     pdf_path = "path_to_your_pdf.pdf"  # Replace with your actual PDF file path
#     documents = extract_text_from_pdf_to_documents(pdf_path)
    
#     # Generate result using the documents
#     result = generate_result(query, documents)
#     print("Final Answer:", result)
	