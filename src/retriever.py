# --- src/retriever.py ---
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class ComplaintRetriever:
    def __init__(self, 
                 index_path: str = "vector_store/complaints.index", 
                 metadata_path: str = "vector_store/metadata.pkl",
                 tfidf_path: str = "vector_store/tfidf.pkl"):
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Load FAISS index
        self.index = self._load_index(index_path)
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_path)
        self._validate_data_integrity()
        
        # Load TF-IDF for hybrid search
        self.tfidf_vectorizer = self._load_tfidf(tfidf_path)
        self.tfidf_matrix = self._load_tfidf_matrix("vector_store/tfidf_matrix.npz")
    
    def _load_index(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Index file not found at {path}")
        return faiss.read_index(str(path))
    
    def _load_metadata(self, path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        required_cols = {'Product', 'cleaned_text', 'Complaint ID', 'Date received'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Metadata missing required columns: {missing}")
        return data
    
    def _load_tfidf(self, path: str):
        if Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    
    def _load_tfidf_matrix(self, path: str):
        if Path(path).exists():
            return np.load(path)['matrix']
        return None
    
    def _validate_data_integrity(self):
        if len(self.metadata) != self.index.ntotal:
            print(f"Warning: Truncating to min size (metadata: {len(self.metadata)}, index: {self.index.ntotal})")
            min_size = min(len(self.metadata), self.index.ntotal)
            self.metadata = self.metadata.iloc[:min_size]
            self.index = self.index[:min_size]
    
    def search(self, query: str, k: int = 5, product_filter: Optional[str] = None, 
               min_score: float = 0.4, date_range: Optional[tuple] = None) -> List[Dict]:
        
        # Semantic search
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k*3)  # Get extra for filtering
        
        # Hybrid search with TF-IDF if available
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_scores = np.squeeze(query_tfidf.dot(self.tfidf_matrix.T).toarray())
        else:
            tfidf_scores = np.zeros(len(self.metadata))
        
        results = []
        for idx, semantic_score in zip(indices[0], distances[0]):
            if semantic_score < min_score:
                continue
                
            record = self.metadata.iloc[idx]
            
            # Apply filters
            if product_filter and record['Product'] != product_filter:
                continue
                
            if date_range:
                try:
                    complaint_date = datetime.strptime(record['Date received'], '%Y-%m-%d')
                    if not (date_range[0] <= complaint_date <= date_range[1]):
                        continue
                except (KeyError, ValueError):
                    continue
            
            # Combine semantic and keyword scores
            hybrid_score = 0.7 * semantic_score + 0.3 * tfidf_scores[idx]
            
            results.append({
                'text': record['cleaned_text'],
                'product': record['Product'],
                'issue': record.get('Issue', 'N/A'),
                'date': record.get('Date received', 'Unknown'),
                'score': float(hybrid_score),
                'id': record['Complaint ID']
            })
        
        # Return top-k results
        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]
    
    def batch_search(self, queries: List[str], k: int = 3, **kwargs) -> Dict[str, List[Dict]]:
        query_embeddings = self.model.encode(queries)
        distances, indices = self.index.search(query_embeddings, k*2)  # Get extra
        
        results = {}
        for i, query in enumerate(queries):
            query_results = []
            for idx, score in zip(indices[i], distances[i]):
                if score < kwargs.get('min_score', 0.4):
                    continue
                    
                record = self.metadata.iloc[idx]
                if kwargs.get('product_filter') and record['Product'] != kwargs['product_filter']:
                    continue
                    
                query_results.append({
                    'text': record['cleaned_text'],
                    'product': record['Product'],
                    'score': float(score),
                    'id': record['Complaint ID']
                })
            
            results[query] = sorted(query_results, key=lambda x: x['score'], reverse=True)[:k]
        
        return results
    
    def get_statistics(self) -> Dict:
        return {
            'total_complaints': len(self.metadata),
            'products': self.metadata['Product'].value_counts().to_dict(),
            'common_issues': self.metadata['Issue'].value_counts().head(10).to_dict()
        }

def retrieve_context(question: str, k: int = 5, product_filter: Optional[str] = None) -> List[Dict]:
    retriever = ComplaintRetriever()
    
    # Query expansion for better retrieval
    expanded_queries = generate_query_variations(question)
    
    # Search across all variations
    all_results = []
    for query in expanded_queries:
        results = retriever.search(
            query,
            k=k,
            product_filter=product_filter,
            min_score=0.4
        )
        all_results.extend(results)
    
    # Deduplicate and get top results
    unique_results = {r['id']: r for r in all_results}.values()
    return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:k]

def generate_query_variations(query: str) -> List[str]:
    """Generate multiple query variations for better retrieval"""
    variations = [query]
    
    # Add product-specific variations if detected
    products = ['credit card', 'loan', 'BNPL', 'savings', 'money transfer']
    for product in products:
        if product in query.lower():
            variations.extend([
                f"common complaints about {product}",
                f"issues with {product} services",
                f"problems customers have with {product}"
            ])
    
    # Add time-related variations if detected
    time_terms = ['recent', 'last month', 'this year', '2024']
    if any(term in query.lower() for term in time_terms):
        variations.append(f"recent {query}")
    
    return variations