from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader  # Ensure this is available

DATA_PATH = "./knowledgeBase/"

def load_documents():
    # Create a DirectoryLoader instance with the specified path and loader class
    loader = PyPDFLoader(DATA_PATH+"/syllabus.pdf",)
    
    # Load documents from the specified directory
    documents = loader.load()
    
    return documents
 