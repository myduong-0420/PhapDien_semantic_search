import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

# Load model
class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        # Convert the list of texts to embeddings
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text):
        # Convert a single query to its embedding
        return self.model.encode(text).tolist()

# Instantiate wrapper with model
embedding_model = SentenceTransformerWrapper('bkai-foundation-models/vietnamese-bi-encoder')

# Load vector store
vector_db = Chroma(
    persist_directory="chroma_db_new",
    embedding=embedding_model  # Use your SentenceTransformerWrapper instance
)

# Display results
def retrieve_info(query, k):
    results = vector_db.similarity_search(query, k=5)
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...")  # Display a preview of the chunk
        return f"Result {i+1}:\nMetadata: {doc.metadata}\nContent: {doc.page_content[:200]}..."
    
demo = gr.Interface(
    fn=retrieve_info,
    inputs=["text", gr.Number(default=5, label="k (Number of chunks to retrieve)")],
    outputs=[gr.Textbox(label="Output chunk(s)", lines=500)],
)

demo.launch()
