import os
# Requires: pip install docx2txt
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths - Assuming you put your Word docs in a new folder
BOOK_DIR = "./data/Tai Ji Book Final Draft"
CHROMA_DB_DIR = "./data/chroma_db"

def add_english_book():
    print("Step 1: Loading Word documents (.docx)...")
    # This loader specifically targets .docx files in the folder
    loader = DirectoryLoader(
        BOOK_DIR, 
        glob="*.docx", 
        loader_cls=Docx2txtLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} Word documents.")

    print("\nStep 2: Chunking the English text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        # Notice we added ". " to handle English sentence splitting, 
        # while keeping the Chinese punctuation for safety!
        separators=["\n\n", "\n", ". ", "。", "，", " "] 
    )
    new_chunks = text_splitter.split_documents(documents)
    print(f"Split book into {len(new_chunks)} individual chunks.")

    print("\nStep 3: Initializing the SAME Multilingual Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'} 
    )

    print("\nStep 4: Appending to existing Vector Database...")
    # Load the existing database you made from the Chinese transcripts
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model
    )
    
    # Add the new English book chunks into the same space
    vector_db.add_documents(documents=new_chunks)
    
    print("\nSuccess! English book has been merged into the Taichi database.")

if __name__ == "__main__":
    add_english_book()