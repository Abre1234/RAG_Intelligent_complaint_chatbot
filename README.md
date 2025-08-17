# RAG Intelligent Complaint Chatbot  

This project is a **Naive Retrieval-Augmented Generation (RAG) Chatbot** that answers user queries based on complaint-related documents.  
It uses **LangChain**, **FAISS**, and **OpenAI embeddings** to retrieve relevant information and generate grounded responses.  

## 🚀 Features
- Document ingestion (PDF, text, web data)
- Embedding with OpenAI/HuggingFace
- Vector database using FAISS
- Retrieval-based Q&A
- Interactive interface with Gradio  

## 📂 Project Structure
- `ingest.py` → Load and chunk documents  
- `store.py` → Generate embeddings & store in FAISS  
- `app.py` → Gradio chatbot interface  

## 🔧 Installation
```bash
git clone https://github.com/Abre1234/RAG_Intelligent_complaint_chatbot.git
cd RAG_Intelligent_complaint_chatbot
pip install -r requirements.txt
