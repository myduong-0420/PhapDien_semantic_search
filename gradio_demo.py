import gradio as gr
from sentence_transformers import SentenceTransformer

def greet(name, k):
    return ("Hello, " + name + "!") * k

demo = gr.Interface(
    fn=greet,
    inputs=["text", gr.Slider(value=5, minimum=1, maximum=50, step=1)],
    outputs=[gr.Textbox(label="greeting", lines=500)],
)

demo.launch()
