import os
# You will need to install these: pip install chromadb langchain langchain-huggingface sentence-transformers
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

# Paths to your data
TRANSCRIPT_DIR = "./data/cleaned_transcripts"
CHROMA_DB_DIR = "./data/chroma_db"

def build_database():
    print("Step 1: Loading transcripts...")
    # This loads all .txt files in the directory and automatically attaches 
    # the filename as metadata so we know where each chunk came from!
    loader = DirectoryLoader(
        TRANSCRIPT_DIR, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'autodetect_encoding': True}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} transcript files.")

    print("\nStep 2: Chunking the text...")
    # 500 characters is roughly a good paragraph of Chinese text.
    # The 50 character overlap ensures that if a sentence spans a chunk boundary, 
    # context isn't lost.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "] # Respect Chinese punctuation
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split text into {len(chunks)} individual chunks.")

    print("\nStep 3: Initializing Multilingual Embedding Model...")
    # This specific model maps 50+ languages into the same vector space.
    # English queries will perfectly match Chinese text chunks!
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': device} 
    )

    print("\nStep 4: Creating Vector Database (This will take a few minutes)...")
    # This embeds all the chunks and saves them to a local folder.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"\nSuccess! Database created and saved to {CHROMA_DB_DIR}")
    
    print("\n--- Running a Test Query ---")
    # Notice we can query in English!
    query = "How do I shift my weight and relax?" 
    results = vector_db.similarity_search_with_score(query, k=2) # Get top 2 results
    
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} (Score: {score}):")
        # Shows the source file (e.g., lecture_4.txt)
        print(f"Source: {doc.metadata['source']}") 
        print(f"Text: {doc.page_content}")

if __name__ == "__main__":
    build_database()