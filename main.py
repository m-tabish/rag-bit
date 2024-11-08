__import__("pysqlite3")
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
from dotenv import load_dotenv 
from generate_result import generate_result  # Assuming the function is available in this file
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()



# Constants


CHROMA_PATH = "chroma"
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to convert extracted text into Document objects
def extract_text_to_documents(file_bytes):
    '''Converts text from PDF to Document objects'''
    pdf_reader = PdfReader(BytesIO(file_bytes))
    documents = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            documents.append(Document(page_content=text))
    return documents

# Main function for generating embeddings and searching results
def main(user_input, documents):
    '''----------------------------------------------------------''' 
    # Perform similarity search using the documents (assuming generate_result works with Document objects)
    result = generate_result(user_input, documents)
    return result 
    '''----------------------------------------------------------'''

# Streamlit app title
st.markdown("""<h1 style="text-align:center; font-family:monospace;">RAGBOT</h1>""", unsafe_allow_html=True)

# Multi-line text input for user query
user_input = st.text_area("What would you like to search?")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "png", "jpg", "jpeg", "csv"])

# Variable to store the extracted text
documents = []

# Process the file once it is uploaded
if uploaded_file is not None:
    try:
        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            documents = extract_text_to_documents(file_bytes)
            st.write(f"Extracted text from {len(documents)} pages.")

        # Handle text files (txt)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            documents = [Document(page_content=text)]
            st.write("Extracted text (first 100 characters):", text[:100])

        # You can add other file types (like image, CSV, etc.) here if necessary

        # If there is extracted text, perform the main processing
        if documents:
            st.write("Processing your query...")
            with st.spinner('Generating results...'):
                result = main(user_input, documents)  # Generate results from the query
            
            st.write("Answer to your query:")
            st.write(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer for the app
st.markdown(
    """
    <div style="text-align: center;">
        <h3>Developed by <a href="https://github.com/m-tabish" target="_blank" style="color: #00FF00; display:inline;">Tabish</a></h3> 
    </div>
    """,
    unsafe_allow_html=True,
)

