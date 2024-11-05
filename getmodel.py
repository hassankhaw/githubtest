import os
from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings.sentence_transformer import (
#  SentenceTransformerEmbeddings,
# )
# Define the model name
model_name = "all-MiniLM-L6-v2"

# Create an instance of the SentenceTransformerEmbeddings class
embedding_function = SentenceTransformer(model_name)

# Save the model to a file
model_path = os.path.join("/tmp", model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
embedding_function.save(model_path) 