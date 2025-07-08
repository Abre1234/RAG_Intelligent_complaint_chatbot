# app/app.py
import gradio as gr
from src.rag_pipeline import generate_answer

def run_app():
    def respond(question):
        answer, sources = generate_answer(question)
        return f"{answer}\n\nSources:\n" + "\n".join([f"- {s[:200]}..." for s in sources])
    
    iface = gr.Interface(
        fn=respond,
        inputs=gr.Textbox(label="Question"),
        outputs=gr.Textbox(label="Answer"),
        title="Complaint Analyst"
    )
    iface.launch()