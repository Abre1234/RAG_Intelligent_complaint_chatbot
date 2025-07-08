# src/rag_pipeline.py
from transformers import pipeline
from src.retriever import retrieve_context

qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

def generate_answer(question):
    context = retrieve_context(question)
    prompt = f"""
You are a helpful financial analyst at CrediTrust.
Use the context below to answer the user's question.

Context:
{'\n'.join(context)}

Question: {question}
Answer:
"""
    response = qa_pipeline(prompt, max_length=256, do_sample=True)
    return response[0]['generated_text'], context
