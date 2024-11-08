# RAG-Bot

RAG-Bot is a Retrieval-Augmented Generation (RAG) chatbot designed to provide informative responses based on user queries. It leverages advanced natural language processing techniques to retrieve relevant information from documents and generate coherent answers.

## Features

- **Document Retrieval**: Efficiently retrieves relevant documents based on user input.
- **Natural Language Processing**: Utilizes state-of-the-art models for understanding and generating responses.
- **User-Friendly Interface**: Simple interaction model for users to ask questions and receive answers.
- **Customizable**: Easily extendable to include additional features or integrate with other data sources.

## Installation

To set up RAG-Bot locally, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps to Install

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/m-tabish/rag-bot.git
   cd rag-bot

   ```

2. Create a Virtual Environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use  `venv\Scripts\activate`
   pip install --quiet --upgrade langchain langchain-community langchain-chroma PyPDF2 langchain_google_genai streamlit

   ```

3. Install Required Packages:

   ```bash
   pip install -r requirements.txt

   ```

4. Set Up Environment Variables:
   Create a .env file in the project root directory and add your API keys and other configuration settings as required by the application.

### Usage

To run the RAG-Bot, execute the following command in your terminal:

```bash
streamlit run main.py

```
