# --- src/rag_pipeline.py ---

from transformers import pipeline
from src.retriever import retrieve_context

qa_pipeline = pipeline("text-generation", model="distilgpt2")



def generate_answer(question):
    context = retrieve_context(question)
    prompt = f"""
You are a helpful financial analyst at CrediTrust.
Use the context below to answer the user's question.

Context:
{chr(10).join(context)}

Question: {question}
Answer:
"""
    response = qa_pipeline(prompt, max_length=256, do_sample=True)
    return response[0]['generated_text'], context

if __name__ == "__main__":
    question = "Why are customers complaining about Buy Now, Pay Later?"
    answer, context = generate_answer(question)

    print("\n--- Question ---")
    print(question)

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Retrieved Context Chunks ---")
    for i, chunk in enumerate(context, 1):
        print(f"[{i}] {chunk}")
 
