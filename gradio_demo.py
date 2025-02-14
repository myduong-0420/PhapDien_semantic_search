import os
import zipfile
from huggingface_hub import hf_hub_download
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# Step 1: Download and Extract the Chroma Vector Store
def prepare_chroma_db(hf_token=None):
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        print("Downloading chroma_db.zip from the dataset repository...")
        zip_path = hf_hub_download(
            repo_id="camiellia/phapdien_demo",  # dataset repository
            repo_type="dataset",
            filename="chroma_db.zip",
            token=hf_token
        )
        print(f"Downloaded to {zip_path}")
        
        # Extract the zip file into the persist_directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(persist_directory)
        print(f"Extracted chroma_db to ./{persist_directory}")
    else:
        print(f"{persist_directory} directory already exists.")
    return persist_directory

persist_directory = prepare_chroma_db()

# Step 2: wrapper
class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        # Convert the list of texts to embeddings
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text):
        # Convert a single query to its embedding
        return self.model.encode(text).tolist()

embedding_model = SentenceTransformerWrapper('bkai-foundation-models/vietnamese-bi-encoder')

# Step 3: Load the vector store from the directory
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model  # Use your SentenceTransformerWrapper instance
)

# Step 4: Gradio function
def retrieve_info(query, k):
    results = vector_db.similarity_search(query, k)
    output = ""
    for i, doc in enumerate(results):
        output += f"Result {i+1}:\nMetadata: {doc.metadata}\nContent: {doc.page_content[:1000]}\n\n"
    return output

# Step 5: Launch the Gradio interface
demo = gr.Interface(
    fn=retrieve_info,
    inputs=["text", gr.Number(label="k (Number of chunks to retrieve)")],
    outputs=[gr.Textbox(label="Output chunk(s)", lines=25)],
)

demo.launch()
