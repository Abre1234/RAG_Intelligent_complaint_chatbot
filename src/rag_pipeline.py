# src/rag_pipeline.py
from transformers import pipeline, AutoTokenizer
from src.retriever import retrieve_context
from src.utils import clean_response_text
import torch
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "google/gemma-2b-it"
MAX_TOKENS = 512
TEMPERATURE = 0.3

@lru_cache(maxsize=1)
def load_model():
    """Cache-loaded model for better performance"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        qa_pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            tokenizer=tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info(f"Loaded model {MODEL_NAME}")
        return qa_pipeline
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def format_prompt(question: str, context_chunks: list) -> str:
    """Create optimized prompt for financial complaint analysis"""
    context_str = "\n".join(
        f"Complaint {i+1} (Product: {chunk['product']}): {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    )
    
    return f"""<s>[INST] <<SYS>>
You are a financial analyst at CrediTrust. Analyze these customer complaints:

{context_str}

Guidelines:
- Be concise but specific
- Reference complaint details when possible
- If unsure, say "I don't have enough information"
- Never make up information
<</SYS>>

Question: {question}[/INST] Answer:"""

def generate_answer(question: str, product_filter: str = None) -> tuple:
    """
    Generate answer to financial complaint question
    Returns:
        tuple: (answer_text, list_of_source_chunks)
    """
    if not question.strip():
        return "Please enter a valid question.", []
    
    try:
        # Retrieve relevant context
        context_chunks = retrieve_context(question, product_filter=product_filter)
        if not context_chunks:
            return "No relevant complaints found.", []
        
        # Load model (cached)
        qa_pipeline = load_model()
        
        # Generate prompt
        prompt = format_prompt(question, context_chunks)
        
        # Generate response
        response = qa_pipeline(
            prompt,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        # Extract and clean answer
        full_response = response[0]['generated_text']
        answer = full_response.split("Answer:", 1)[-1].strip()
        cleaned_answer = clean_response_text(answer)
        
        return cleaned_answer, context_chunks
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error processing your question: {str(e)}", []