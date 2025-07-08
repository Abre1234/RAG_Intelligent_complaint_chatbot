# src/embeddings.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

# 1. Load cleaned data
df = pd.read_csv("data/Processed_data/cleaned_complaints.csv")
texts = df['cleaned_text'].tolist()

# 2. Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

# 3. Generate embeddings in batches (memory-efficient)
batch_size = 1000
embeddings = []
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch = texts[i:i+batch_size]
    embeddings.append(model.encode(batch, show_progress_bar=False))
embeddings = np.vstack(embeddings)

# 4. Create FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product similarity
index.add(embeddings)

# 5. Save index and metadata
faiss.write_index(index, "vector_store/complaints.index")
df.to_pickle("vector_store/metadata.pkl")  # Preserves all columns