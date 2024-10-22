import streamlit as st
import requests
import json
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from ollama import Client

def extract_context(query, collection_name="my_collection", n_results=3):
    """
    Query ChromaDB to get relevant context based on the user's query
    """
    try:
        # Initialize ChromaDB HTTP client
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Get the collection
        collection = chroma_client.get_collection(name=collection_name)
        
        # Perform similarity search
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract and join the documents
        if results and results['documents']:
            context = "\n".join(results['documents'][0])
            return context
        return ""
        
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return ""

def create_super_prompt(query, context):
    """
    Create a super prompt combining the context and user query
    """
    return f"""You are an expert consultant helping executive advisors to get relevant information from internal documents.

            Generate your response by following the steps below:
            1. Recursively break down the question into smaller questions.
            2. For each question/directive:
            2a. Select the most relevant information from the context in light of the conversation history.
            3. Generate a draft response using selected information.
            4. Remove duplicate content from draft response.
            5. Generate your final response after adjusting it to increase accuracy and relevance.
            6. Do not try to summarise the answers, explain it properly.
            6. Only show your final response! 

            Constraints:
            1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT.
            2. Don't mention that you are not able to find the answer in the provided context.
            3. Don't make up the answers by yourself.
            4. Try your best to provide answer from the given context.

            CONTENT:
            {context}

            Based on the above context, please provide the answer to the following question.
            Question: {query}

            Answer:"""

def get_ollama_response(prompt, model="qwen2.5:32b"):
    """
    Get response from Ollama API
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot with RAG", page_icon="🤖")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display title
st.title("🤖 Ollama Chatbot with ChromaDB Context")

# Sidebar configurations
st.sidebar.title("Settings")
model = st.sidebar.selectbox(
    "Choose a model",
    ["qwen2.5:32b", "llama3.2", "llama3.1", "llama3.1:70b"]
)

collection_name = st.sidebar.text_input(
    "ChromaDB Collection Name",
    value="col1"
)

n_results = st.sidebar.slider(
    "Number of context results",
    min_value=1,
    max_value=5,
    value=3
)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating response..."):
           # First, get relevant context from ChromaDB
            context = extract_context(prompt, collection_name, n_results)
            
            # Create super prompt with context
            super_prompt = create_super_prompt(prompt, context)
            
            # Debug information (can be toggled in sidebar)
            if st.sidebar.checkbox("Show debug info"):
                st.info("Retrieved Context:")
                st.code(context)
                st.info("Super Prompt:")
                st.code(super_prompt)
            
            # Get response from Ollama
            response = get_ollama_response(super_prompt, model)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})