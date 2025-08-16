import gradio as gr
from transformers import pipeline
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).parent.parent
VECTOR_STORE = BASE_DIR / "vector_store"
DATA_FILE = BASE_DIR / "data" / "Processed_data" / "cleaned_complaints.csv"

# ============== RESOURCE LOADING ==============
def load_resources():
    """Load all required resources with error handling"""
    try:
        # 1. Load the sentence transformer model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Load the FAISS index
        print("Loading FAISS index...")
        index = faiss.read_index(str(VECTOR_STORE / "complaints.index"))
        
        # 3. Load metadata
        print("Loading metadata...")
        metadata = pd.read_pickle(VECTOR_STORE / "metadata.pkl")
        texts = metadata['cleaned_text'].tolist()
        
        # 4. Initialize QA pipeline
        print("Initializing QA pipeline...")
        qa_pipeline = pipeline("text-generation", model="distilgpt2")
        
        return model, index, texts, qa_pipeline
        
    except Exception as e:
        print(f"\nERROR: Failed to load resources: {str(e)}")
        print("\nPlease verify:")
        print(f"1. File exists: {VECTOR_STORE / 'complaints.index'}")
        print(f"2. File exists: {VECTOR_STORE / 'metadata.pkl'}")
        print(f"3. You have internet access to download models")
        print("\nTry regenerating vector store if needed:")
        print("python src/chunking_embedding.py")
        sys.exit(1)

# Load all resources
print("Initializing system...")
retriever_model, retriever_index, texts, qa_pipeline = load_resources()
print("All resources loaded successfully!")

# ============== CORE FUNCTIONS ==============
def generate_answer(question):
    """Generate answer using RAG pipeline"""
    try:
        # 1. Retrieve relevant context
        question_embedding = retriever_model.encode([question])
        distances, indices = retriever_index.search(question_embedding.astype(np.float32), 5)
        contexts = [texts[i] for i in indices[0]]
        
        # 2. Generate prompt
        prompt = f"""Analyze these customer complaints:
        {' '.join(contexts)}
        
        Question: {question}
        Answer:"""
        
        # 3. Generate response
        response = qa_pipeline(
            prompt,
            max_length=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256
        )
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        
        # 4. Format sources
        sources = "\n\n".join(
            f"• {ctx[:200]}..." if len(ctx) > 200 else f"• {ctx}"
            for ctx in contexts
        )
        
        return answer, sources
        
    except Exception as e:
        return f"Error generating answer: {str(e)}", "No sources available"

# ============== GRADIO INTERFACE ==============
with gr.Blocks(title="CrediTrust Complaint Analyst") as app:
    gr.Markdown("# 📊 CrediTrust Complaint Analyst")
    gr.Markdown("Transform customer feedback into actionable insights")
    
    with gr.Row():
        question = gr.Textbox(
            label="Ask about customer complaints",
            placeholder="E.g. What are common issues with credit cards?",
            lines=3
        )
    
    with gr.Row():
        submit_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")
    
    with gr.Row():
        answer = gr.Textbox(
            label="Analysis", 
            interactive=False,
            lines=5
        )
    
    with gr.Accordion("View sources used", open=False):
        sources = gr.Textbox(
            label="Relevant Complaints", 
            lines=5,
            max_lines=15
        )
    
    def clear_form():
        return "", ""
    
    submit_btn.click(
        fn=generate_answer,
        inputs=question,
        outputs=[answer, sources]
    )
    
    clear_btn.click(
        fn=clear_form,
        outputs=[question, answer]
    )

if __name__ == "__main__":
    app.launch()