from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

Data_path = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    # Load documents from the specified directory
    loader = DirectoryLoader(Data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings using a Hugging Face model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a FAISS vector store from the text chunks and embeddings
    db = FAISS.from_documents(text_chunks, embeddings)
    
    # Save the FAISS vector store locally
    db.save_local(DB_FAISS_PATH)
if __name__ == '__main__':
    create_vector_db()
