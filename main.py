import os
from dotenv import load_dotenv
from generate_result import generate_result
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
KNOWLEDGE_BASE = os.path.abspath("./knowledgeBase")
PDF_FILE_NAME = "attention-is-all-you-need.pdf"  # Replace with your PDF file name
PDF_FILE_PATH = os.path.join(KNOWLEDGE_BASE, PDF_FILE_NAME)

# Ensure the knowledge base directory exists
os.makedirs(KNOWLEDGE_BASE, exist_ok=True)

# Function to process and extract text from a PDF file and return Document objects
def extract_text_from_pdf_to_documents(pdf_path):
    documents = []
    try:
        # Check if the PDF file exists before trying to open it
        if not os.path.isfile(pdf_path):
            st.error(f"The file {pdf_path} does not exist.")
            return documents
        
        with open(pdf_path, "rb") as file:  
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Create a Document object for each page with the extracted text
                    doc = Document(page_content=text)
                    documents.append(doc)
    except PermissionError:
        st.error(f"Permission denied when trying to access {pdf_path}. Please check your permissions.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return documents

# Main function for generating embeddings and searching results
def main(user_input, documents):
    '''----------------------------------------------------------''' 
    # Perform similarity search using the documents (assuming generate_result works with Document objects)
    result = generate_result(user_input, documents)
    return result 
    '''----------------------------------------------------------'''

# Streamlit app title
st.markdown("""<h1 style="text-align:center; font-family:monospace;">Answer Bot 🤖</h1>""", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size:24px">
        <h5>The bot take the knowledge from "Attention is all you need" Research Paper</h5> 
    </div>
    """,
    unsafe_allow_html=True,
)
# Multi-line text input for user query
user_input = st.text_area("What would you like to search?")

# Variable to store the extracted text from the specified PDF file in KNOWLEDGE_BASE
documents = extract_text_from_pdf_to_documents(PDF_FILE_PATH)

# Process user input if any documents are available
if user_input and documents:
    result = main(user_input, documents)
    st.write(result)

# Footer for the app
st.markdown(
    """
    <div style="text-align: center;">
        <h3>Developed by <a href="https://github.com/m-tabish" target="_blank" style="color: #00FF00; display:inline;">Tabish</a></h3> 
    </div>
    """,
    unsafe_allow_html=True,
)