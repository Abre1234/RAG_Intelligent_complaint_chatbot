import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import gc

class DataPreprocessor:
    def __init__(self):
        self.raw_path = Path("data/Raw_data/raw_complaints.csv")
        self.processed_dir = Path("data/Processed_data")
        
        # Create processed folders
        self.processed_dir.mkdir(exist_ok=True)
        (self.processed_dir / "chunks").mkdir(exist_ok=True)
    
    def clean_text(self, text):
        """Clean individual complaint text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    def process_in_chunks(self, chunk_size=10000):
        """Process data in memory-friendly chunks"""
        # Target products from your project brief
        target_products = [
            'Credit card', 
            'Personal loan',
            'Payday loan',
            'Student loan',
            'Money transfers'
        ]
        
        # Initialize empty list for chunks
        processed_chunks = []
        
        # Read CSV in chunks
        chunk_reader = pd.read_csv(
            self.raw_path,
            chunksize=chunk_size,
            dtype={'Product': 'category'},
            usecols=['Product', 'Issue', 'Consumer complaint narrative', 'Complaint ID']
        )
        
        for i, chunk in enumerate(tqdm(chunk_reader, desc="Processing chunks")):
            # Filter for target products
            chunk = chunk[chunk['Product'].isin(target_products)]
            
            # Remove empty narratives
            chunk = chunk[chunk['Consumer complaint narrative'].notna()]
            
            # Clean text
            chunk['cleaned_text'] = chunk['Consumer complaint narrative'].apply(self.clean_text)
            
            # Save each processed chunk
            chunk_path = self.processed_dir / f"chunks/processed_chunk_{i}.parquet"
            chunk.to_parquet(chunk_path)
            processed_chunks.append(chunk_path)
            
            # Clear memory
            del chunk
            gc.collect()
        
        # Combine all chunks
        self._combine_chunks(processed_chunks)
    
    def _combine_chunks(self, chunk_files):
        """Combine processed chunks into final dataset"""
        combined = pd.concat([pd.read_parquet(f) for f in chunk_files])
        combined.to_csv(self.processed_dir / "cleaned_complaints.csv", index=False)
        print(f"Saved cleaned data to {self.processed_dir/'cleaned_complaints.csv'}")
        return combined

if __name__ == "__main__":
    print("Starting data preprocessing...")
    processor = DataPreprocessor()
    processor.process_in_chunks()
    print("Data preprocessing completed!")