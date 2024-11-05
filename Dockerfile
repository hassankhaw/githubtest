# FROM python:3
# FROM public.ecr.aws/lambda/python:3.12
# FROM python:latest
FROM python:3.12-bookworm

WORKDIR /usr/src/app

# Install Rust and Cargo
# RUN apt-get update 
RUN curl https://sh.rustup.rs -sSf -y | sh
# RUN apt-get install -y cargo
# RUN python --version

# Install dependencies
# RUN pip install onnxruntime
# RUN sudo apt-get install python3-dev
# RUN python3 -m pip install --upgrade pip
# RUN sudo apt install build-essential
# RUN python3 -m pip install hnswlib
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN pip install --upgrade pip
RUN pip install chromadb
RUN pip install --no-cache-dir langchain_huggingface
RUN pip install --no-cache-dir langchain_chroma
RUN pip install --no-cache-dir langchain_community
RUN pip install --no-cache-dir langchain_text_splitters
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir openai
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir ollama
RUN pip install --no-cache-dir pypdf

ADD all-MiniLM-L6-v2 /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
COPY . .

# RUN python rag.py

# CMD [ "python", "./rag_query.py" ]
CMD ["python", "/usr/src/app/rag_query.py"]
# CMD [ "python", "--version" ]
