from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch


CHROMA_DB_DIR = "./data/chroma_db"

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

vector_db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

print("\n--- Running a Test Query ---")
query = "What is chi?" 
results = vector_db.similarity_search_with_score(query, k=5)

for i, (doc, score) in enumerate(results):
    print(f"\nResult {i+1} (Score: {score}):")
    print(f"Source: {doc.metadata['source']}") 
    print(f"Text: {doc.page_content}")