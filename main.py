import os
import shutil
from dotenv import load_dotenv      
from generate_and_store_embeddings import generate_and_store_embeddings
from langchain_doc_loader import load_documents
from generate_result import generate_result
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma 
CHROMA_PATH = "chroma"
# Load environment variables from .env file
load_dotenv() 

# Create the 'uploaded_files' directory if it doesn't exist
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Main function for generating embeddings and searching results
def main(user_input, text):
    '''----------------------------------------------------------''' 
    # Generate embeddings and store in ChromaDB (if needed)
    # vector_store = generate_and_store_embeddings(text)  # Uncomment if embeddings generation is required
    
    '''----------------------------------------------------------'''
    # Retrieving results (conducting similarity search)
    result = generate_result(user_input, text) 
    return result 
    '''----------------------------------------------------------'''

# Streamlit app title
st.markdown("""
            <h1 style = "text-align:center; font-family:monospace;">RAGBOT</h1>""", unsafe_allow_html=True)

# Multi-line text input for user query
user_input = st.text_area("What would you like to search?:")
 
# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg", "txt", "csv"])

# Variable to store the extracted text
text = ""

# Process the file once it is uploaded
if uploaded_file is not None:
    try:
        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            pdf_reader = PdfReader(BytesIO(file_bytes))
            
            # Read the text content from all pages in the PDF
            text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            st.write("Extracted text (first 100 characters):", text[:100])
        
        # Handle text files (txt)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            st.write("Extracted text (first 100 characters):", text[:100])

        # Other file types like image or CSV can be processed similarly if needed

        # If there is text extracted, perform the main processing
        if text:
            st.write("Processing your query...")
            with st.spinner('Generating results...'):
                result = main(user_input, text)  # Generate results from the query
            
            st.write("Answer to your query:")
            st.write(result) 
            
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
st.markdown(
        """
        <div style="text-align: center;">
            <h3>Developed by <a href="https://github.com/m-tabish" target="_blank" style = "color: #00FF00 ;display:inline;">Tabish</a></h3> 
        </div>
        """,
        unsafe_allow_html=True,
    )


'''
.\Scripts\deactivate
.\Scripts\activate

streamlit run main.py
'''