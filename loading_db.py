#Loading the data stored in DB
from sentence_transformers import SentenceTransformer 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import getpass
import os
from langchain import hub

CHROMA_PATH = 'chroma'

os.environ["GOOGLE_API_KEY"] = getpass.getpass()


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def load_db(query_text):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function= embedding_function)
    
    try:
        results = vector_store.similarity_search_with_relevance_scores(query_text,k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            return
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatVertexAI(model="gemini-1.5-flash") 
        
        example_messages = prompt.invoke(
            {"context": "filler context", "question": "filler question"}
        ).to_messages()
        

        print(example_messages[0].content)
        return context_text
    
    except Exception as e:
        print("Exception in loading and finding similarity ",e)