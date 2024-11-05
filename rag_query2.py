import uuid
import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings.sentence_transformer import (
#  SentenceTransformerEmbeddings,
# )
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
 SentenceTransformerEmbeddings,
)
from ollama import Client
from flask import Flask, request


# load the document and split it into pages
loader = PyPDFLoader("./docs/2404.07143v2.pdf")
pages = loader.load_and_split()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(pages)

# Define the model name
model_name = "all-MiniLM-L6-v2"

# create the open-source embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    cache_folder=".cache/chroma/onnx_models/all-MiniLM-L6-v2"
)

# hostname = "host.docker.internal"
hostname = "localhost"

# create the chroma client
client = chromadb.HttpClient(host=hostname, port=8000, settings=Settings(allow_reset=True))
# client.reset() # resets the database
collection_name = "col1"
# delete the existing collection if it exists
try:
    client.delete_collection(collection_name)
except:
    pass
# create the new collection or get it if it already exists
collection = client.get_or_create_collection(collection_name)
# collection = client.create_collection(collection_name)
for doc in docs:
    collection.add(
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )

# tell LangChain to use our client and collection name
db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
# query = "What training does the model have?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# When a query is sent to the Flask app, the Extract_context function is called. 
# This function performs a similarity search on the ChromaDB collection, 
# retrieving relevant documents based on the query.ollama  
# retrieving relevant documents based on the query.ollama thank 
def Extract_context(query):
    chroma_client = chromadb.HttpClient(host=hostname, port=8000,settings=Settings(allow_reset=True))
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
    docs = db.similarity_search(query)
    fullcontent =''
    for doc in docs:
        fullcontent ='. '.join([fullcontent,doc.page_content])

    return fullcontent


def get_system_message_rag(content):
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
{content}
"""

def get_ques_response_prompt(question):
 return f"""
Based on the above context, please provide the answer to the following question:
{question}
"""

def generate_rag_response(context, query):
    #combined_query = f"{query} {context}"
    prompt = f"Please answer the following question based on the context provided: {query}. Context: {context}. Please use the information from the context to inform your answer."
    client = Client(host='http://'+hostname+':11434')
    response = client.chat(model="llama3.1", messages=[
        {"role": "user", "content": prompt}
    ])
  
    #print(content)
    #print(query)
    return response['message']['content']

app = Flask(__name__)
@app.route('/query', methods=['POST'])
def respond_to_query():
    if request.method == 'POST':
        data = request.get_json()
        # Assuming the query is sent as a JSON object with a key named 'query'
        query = data.get('query')
        # Here you can process the query and generate a response
        # response = f'This is the response to your query:\n {get_reponse(query)}'
        context = Extract_context(query)
        response = generate_rag_response(context, query)
        return response
        # return "context"
 
if __name__ == '__main__':
 app.run(debug=True, host='0.0.0.0', port=5001)