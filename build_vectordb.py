import os
import shutil
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
TRANSCRIPT_DIR = "./data/cleaned_transcripts"
BOOK_DIR = "./data/Tai Ji Book Final Draft"
CHROMA_DB_DIR = "./data/chroma_db"

def clear_old_database():
    """Deletes the old database folder if it exists to prevent duplicates."""
    if os.path.exists(CHROMA_DB_DIR):
        print("Deleting old database...")
        shutil.rmtree(CHROMA_DB_DIR)
        print("Old database cleared.")
    else:
        print("No existing database found to delete.")

def get_chunks():
    text_splitter_transcripts = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "。", "，", " "]
    )

    text_splitter_books = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "。", "，", " "]
    )
    
    all_chunks = []

    # Load and chunk the Chinese Transcripts
    print("\nLoading Chinese Transcripts...")
    if os.path.exists(TRANSCRIPT_DIR):
        transcript_loader = DirectoryLoader(
            TRANSCRIPT_DIR, 
            glob="*.txt",
            loader_cls=TextLoader, 
            loader_kwargs={'autodetect_encoding': True}
        )
        transcript_docs = transcript_loader.load()
        transcript_chunks = text_splitter_transcripts.split_documents(transcript_docs)
        all_chunks.extend(transcript_chunks)
        print(f"Added {len(transcript_chunks)} transcript chunks.")
    else:
        print(f"Warning: {TRANSCRIPT_DIR} not found.")

    # Load and chunk the English Book
    print("\nLoading English Book...")
    if os.path.exists(BOOK_DIR):
        book_loader = DirectoryLoader(
            BOOK_DIR, 
            glob="*.docx", 
            loader_cls=Docx2txtLoader
        )
        book_docs = book_loader.load()
        book_chunks = text_splitter_books.split_documents(book_docs)
        all_chunks.extend(book_chunks)
        print(f"Added {len(book_chunks)} book chunks.")
    else:
        print(f"Warning: {BOOK_DIR} not found.")

    return all_chunks

def build_database():
    clear_old_database()

    if torch.cuda.is_available():
            device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Loading embeddings on: {device.upper()}")

    embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': device} 
    )

    all_chunks = get_chunks()
    
    # Build the database
    if all_chunks:
        print(f"\nBuilding new Vector Database with {len(all_chunks)} total chunks...")
        print("(This may take a few minutes...)")
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIR
        )
        print(f"\nSuccess! Fresh database saved to {CHROMA_DB_DIR}")
    else:
        print("\nNo data found to build the database.")

    

if __name__ == "__main__":
    build_database()