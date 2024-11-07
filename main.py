import os
import shutil
from dotenv import load_dotenv       
from generate_result import generate_result
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma 

# Defining constants for file paths
CHROMA_PATH = "chroma"
UPLOAD_DIR = "./uploaded_files"

# Loading environment variables from .env file
load_dotenv() 

# Creating the 'uploaded_files' directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Main function for generating embeddings and searching results
def main(user_input, text):
    '''----------------------------------------------------------''' 
    # Generating embeddings and storing them in ChromaDB (if needed)
    # vector_store = generate_and_store_embeddings(text)  # Uncomment if embeddings generation is required
    
    '''----------------------------------------------------------'''
    # Retrieving results by conducting similarity search
    result = generate_result(user_input, text) 
    return result 
    '''----------------------------------------------------------'''

# Setting the title of the Streamlit app
st.markdown("""
            <h1 style="text-align:center; font-family:monospace;">RAGBOT</h1>""", unsafe_allow_html=True)

# Creating a multi-line text input for the user to ask a query
user_input = st.text_area("What would you like to search?:")
 
# Setting up the file uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg", "txt", "csv"])

# Variable to store the extracted text
text = ""

# Processing the uploaded file once it is uploaded
if uploaded_file is not None:
    try:
        # Handling PDF files
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            pdf_reader = PdfReader(BytesIO(file_bytes))
            
            # Extracting text from all pages of the PDF
            text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            st.write("Extracted text (first 100 characters):", text[:100])
        
        # Handling text files (txt)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            st.write("Extracted text (first 100 characters):", text[:100])

        # Processing other file types (e.g., images or CSV) can be added here if needed

        # If text is extracted, perform the main processing
        if text:
            st.write("Processing your query...")
            with st.spinner('Generating results...'):
                result = main(user_input, text)  # Generating results based on user query
            
            st.write("Answer to your query:")
            st.write(result) 
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Displaying developer information at the bottom of the page
st.markdown(
        """
        <div style="text-align: center;">
            <h3>Developed by <a href="https://github.com/m-tabish" target="_blank" style="color: #00FF00; display:inline;">Tabish</a></h3> 
        </div>
        """,
        unsafe_allow_html=True,
    )


# Instructions for running the Streamlit app
'''
.\Scripts\deactivate
.\Scripts\activate

streamlit run main.py
'''
