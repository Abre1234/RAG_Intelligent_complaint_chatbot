import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

st.title("Complaint Analysis")
pipeline = load_pipeline()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Analyzing..."):
        answer = pipeline.query(query)
    st.write("Answer:", answer)