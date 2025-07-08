# --- src/retriever.py ---

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
import pandas as pd

class ComplaintRetriever:
    def __init__(self, index_path: str = "vector_store/complaints.index", 
                 metadata_path: str = "vector_store/metadata.pkl"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.index = self._load_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self._validate_data_integrity()

    def _load_index(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Index file not found at {path}")
        return faiss.read_index(str(path))

    def _load_metadata(self, path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        required_cols = {'Product', 'cleaned_text', 'Complaint ID'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Metadata missing required columns: {missing}")
        return data

    def _validate_data_integrity(self):
        if len(self.metadata) != self.index.ntotal:
            print(f"Warning: Truncating to min size (metadata: {len(self.metadata)}, index: {self.index.ntotal})")
            min_size = min(len(self.metadata), self.index.ntotal)
            self.metadata = self.metadata.iloc[:min_size]

    def search(self, query: str, k: int = 5, product_filter: Optional[str] = None, min_score: float = 0.4,
               date_range: Optional[tuple] = None) -> List[Dict]:
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k*10 if product_filter or date_range else k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score < min_score:
                continue
            record = self.metadata.iloc[idx]
            if product_filter and record['Product'] != product_filter:
                continue
            if date_range:
                try:
                    complaint_date = datetime.strptime(record['Date received'], '%Y-%m-%d')
                    if not (date_range[0] <= complaint_date <= date_range[1]):
                        continue
                except (KeyError, ValueError):
                    continue
            results.append({
                'text': record['cleaned_text'],
                'product': record['Product'],
                'issue': record.get('Issue', 'N/A'),
                'date': record.get('Date received', 'Unknown'),
                'score': float(score),
                'id': record['Complaint ID']
            })
            if len(results) >= k:
                break
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def batch_search(self, queries: List[str], k: int = 3, **kwargs) -> Dict[str, List[Dict]]:
        query_embeddings = self.model.encode(queries)
        distances, indices = self.index.search(query_embeddings, k)
        return {
            query: [
                self._format_result(idx, score, kwargs)
                for idx, score in zip(indices[i], distances[i])
                if score >= kwargs.get('min_score', 0.4)
            ][:k]
            for i, query in enumerate(queries)
        }

    def _format_result(self, idx: int, score: float, filters: dict) -> Optional[Dict]:
        try:
            record = self.metadata.iloc[idx]
            if filters.get('product_filter') and record['Product'] != filters['product_filter']:
                return None
            return {
                'text': record['cleaned_text'],
                'product': record['Product'],
                'score': float(score),
                **{k: record[k] for k in ['Issue', 'Date received', 'Complaint ID'] if k in record}
            }
        except Exception:
            return None

    def get_statistics(self) -> Dict:
        return {
            'total_complaints': len(self.metadata),
            'products': self.metadata['Product'].value_counts().to_dict(),
            'common_issues': self.metadata['Issue'].value_counts().head(10).to_dict()
        }

# Public interface function

def retrieve_context(question: str, k: int = 5) -> List[str]:
    retriever = ComplaintRetriever()
    results = retriever.search(question, k=k)
    return [r['text'] for r in results]

if __name__ == "__main__":
    retriever = ComplaintRetriever()
    print("\nDataset Statistics:")
    print(retriever.get_statistics())
    print("\nTesting Advanced Search:")
    results = retriever.search(
        "unauthorized charges",
        product_filter="Credit card",
        date_range=('2023-01-01', '2023-12-31'),
        min_score=0.6
    )
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['product']} ({res['date']}, score: {res['score']:.2f})")
        print(f"Issue: {res['issue']}")
        print(res['text'][:200] + "...")
