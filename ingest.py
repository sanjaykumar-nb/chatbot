import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/"

def create_vector_db():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.txt'):
            filepath = os.path.join(DATA_PATH, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # The api_key argument has been removed from here
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("API-compatible vector store created successfully.")

if __name__ == "__main__":
    create_vector_db()