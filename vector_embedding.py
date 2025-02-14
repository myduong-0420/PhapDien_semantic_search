import os
import shutil
import numpy as np
from pathlib import Path
import zipfile
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from huggingface_hub import HfApi

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not in the .env file")

# Wrapper for embedding
class SentenceTransformerWrapper:
    def __init__(self, model_name, batch_size=32):
        self.batch_size = batch_size
        self.accelerator = Accelerator()  # Create an accelerator instance
        self.model = SentenceTransformer(model_name)
        # Move the model to the appropriate device
        self.model.to(self.accelerator.device)

    def embed_documents(self, texts):
        # Create a DataLoader for the texts
        dataloader = DataLoader(texts, batch_size=self.batch_size)
        all_embeddings = []
        # Optionally, prepare the DataLoader with accelerator if needed
        dataloader = self.accelerator.prepare(dataloader)
        
        for batch in tqdm(dataloader, desc="Embedding documents"):
            # SentenceTransformer.encode already supports batching;
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings.tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()
    
# Step 1: Unzip HTML files and set up model
def extract_documents(zip_path, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Documents extracted to {extract_dir}")
    return extract_dir

# Step 2: Clean the HTML files
def load_and_clean_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


# Step 3: Process files in directory and extract plain text
def process_html_files(directory, file_pattern="full_*.html"):
    directory = Path(directory)
    documents, metadata = [], []
    
    html_files = list(directory.glob(file_pattern))
    for file_path in tqdm(html_files, desc="Loading and cleaning documents"):
        text = load_and_clean_html(file_path)
        documents.append(text)
        metadata.append({"file_path": str(file_path)})
    
    print(f"Loaded {len(documents)} documents")
    return documents, metadata


# Step 4: Split text into chunks
def split_documents(documents, metadata, chunk_size=2000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[". ", "\n", "; ", "\t"]
    )
    
    splitted_docs, splitted_metadata = [], []
    for doc, meta in tqdm(zip(documents, metadata), desc="Splitting documents", total=len(documents)):
        chunks = text_splitter.split_text(doc)
        splitted_docs.extend(chunks)
        splitted_metadata.extend([meta] * len(chunks))
    
    return splitted_docs, splitted_metadata


# Step 5: Clean the chunks
def clean_chunks(splitted_docs, splitted_metadata, min_length=50):
    cleaned_docs, cleaned_metadata = [], []
    
    for doc, meta in tqdm(zip(splitted_docs, splitted_metadata), desc="Cleaning text", total=len(splitted_docs)):
        phrases = doc.split("\n")
        for phrase in phrases:
            if len(phrase) > min_length and "    " not in phrase:
                cleaned_docs.append(phrase)
                cleaned_metadata.append(meta)
    
    print(f"Cleaned {len(cleaned_docs)} text chunks.")
    return cleaned_docs, cleaned_metadata


# Step 6: Save to ChromaDB
def save_to_chromadb(
    processed_docs, 
    processed_metadata, 
    embedding_model, 
    persist_directory="./chroma_db", 
    batch_size=1024
):
    """
    Save documents to a Chroma vectorstore in batches.
    
    processed_docs: List of text chunks.
    processed_metadata: Corresponding metadata list.
    embedding_model: An embedding model with a method embed_documents.
    persist_directory: Where the vectorstore will be saved.
    batch_size: Number of documents to process per batch.
    """

    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    vector_store = None
    
    # Process documents in batches.
    for i in tqdm(range(0, len(processed_docs), batch_size), desc="Embedding and saving batches"):
        batch_texts = processed_docs[i : i + batch_size]
        batch_metadata = processed_metadata[i : i + batch_size]
        
        # Compute embeddings for the current batch.
        # batch_embeddings = embedding_model.embed_documents(batch_texts)
        
        # Add the batch to the vectorstore.
        if vector_store is None:
        # Initialize Chroma vector store with the first batch
            vector_store = Chroma.from_texts(
                texts=batch_texts,
                embedding=embedding_model,
                metadatas=batch_metadata,
                persist_directory=persist_directory
        )
        else:
            vector_db.add_texts(
                texts=batch_texts, 
                # batch_embeddings=batch_embeddings,
                embedding=embedding_model,
                metadatas=batch_metadata
        )
    
    # Persist changes to disk.
    print(f"Database saved successfully to {persist_directory}")

    return vector_db


# Main script
if __name__ == "__main__":
    # Configuration
    zip_path = "./documents.zip"
    extract_dir = "./vbpl"
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    
    # Step 1: Extract files
    extract_dir = extract_documents(zip_path, extract_dir)
    
    # Step 2: Initialize embedding model
    embedding_model = SentenceTransformerWrapper(model_name, batch_size=32)
    
    # Step 3: Process files
    documents, metadata = process_html_files(extract_dir)
    
    # Step 4: Split text into chunks
    splitted_docs, splitted_metadata = split_documents(documents, metadata)
    
    # Step 5: Clean the text chunks
    processed_docs, processed_metadata = clean_chunks(splitted_docs, splitted_metadata)
    
    # Step 6: Generate embeddings and save to ChromaDB
    save_to_chromadb(processed_docs, processed_metadata, embedding_model)

    shutil.make_archive("chroma_db", "zip", "./chroma_db")
    print("Vector database archived as chroma_db.zip")

    api = HfApi()
    repo_id = "camiellia/phapdien_demo"
    api.upload_file(
    path_or_fileobj="chroma_db.zip",
    path_in_repo="chroma_db.zip",
    repo_id=repo_id,
    repo_type="dataset",
    token=hf_token,
    )
    print("Uploaded chroma_db.zip to Hugging Face Hub")