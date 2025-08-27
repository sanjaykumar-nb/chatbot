import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
# We are now importing TextLoader specifically
from langchain_community.document_loaders import TextLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/"

def create_vector_db():
    """
    Reads documents from the data directory, splits them into chunks, 
    creates embeddings, and saves them to a FAISS vector store.
    """
    documents = []
    # Loop through all the files in the data directory
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.txt'):
            filepath = os.path.join(DATA_PATH, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            # Each file is loaded into the documents list
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Vector store created successfully and saved locally.")

if __name__ == "__main__":
    create_vector_db()