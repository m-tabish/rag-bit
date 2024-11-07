# load_data.py
from PyPDF2 import PdfReader

def load_data(filename):
    # Create a PDF reader object
    reader = PdfReader(filename)
    pages_content = ""
    
    # Loop through all the pages and extract text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_content += text  # Concatenate text if it's not None
    
    return pages_content  # Return a single string with all text combined
