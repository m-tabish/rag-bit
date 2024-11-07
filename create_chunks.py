from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_doc_loader import load_documents

def create_chunks(documents):
    try:
        # Initializing  the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    
            chunk_overlap=500,
            length_function = len,
            add_start_index = True 
        )
        
        
        # Use split_text on the extracted_data (which should be a single string), 
        # split text works on single text input and split_documents work on document objects
        # Return type is List of Strings
        chunks = text_splitter.split_documents(documents)
        if(chunks):
            with open("./results/chunks.txt", 'w', encoding="utf-8") as f:  # specify UTF-8 encoding 
               f.write(f"Length of chunks {len(chunks)}\n\n")
               for chunk in chunks: #data_chunks is a list of text
                   f.write(f"{chunk}\n\n") 
        else:
            print("Could not write chunks")
        return chunks  # Return the list of text chunks
    except Exception as e:
        print("Error in chunking:", e)
        return []
