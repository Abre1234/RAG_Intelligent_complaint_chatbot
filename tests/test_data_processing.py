# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import clean_text, filter_complaints, chunk_text
import numpy as np

class TestDataProcessing:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'BNPL', 'Savings account', 'Money transfer', 'Mortgage'],
            'Consumer complaint narrative': [
                'I was charged unexpected fees!',
                'The interest rate was higher than promised',
                '',
                '   ',  # whitespace only
                'I cannot access my money transfers',
                'My mortgage application was denied'
            ],
            'Date received': ['2023-01-01'] * 6,
            'Issue': ['Fees'] * 6
        })

    def test_clean_text(self):
        dirty_text = "  I was charged 25% MORE! (unexpected)   "
        clean = clean_text(dirty_text)
        assert clean == "i was charged 25 more unexpected"
        
        # Test empty string
        assert clean_text("") == ""
        
        # Test with newlines and special chars
        assert clean_text("Hello\nWorld! @#$") == "hello world"

    def test_filter_complaints(self, sample_data):
        # Test product filtering
        filtered = filter_complaints(sample_data, [
            'Credit card', 
            'Personal loan', 
            'BNPL', 
            'Savings account', 
            'Money transfer'
        ])
        
        assert len(filtered) == 5  # Mortgage should be removed
        assert 'Mortgage' not in filtered['Product'].values
        
        # Test empty narrative filtering
        filtered = filter_complaints(filtered, None)  # Don't filter by product
        assert len(filtered) == 3  # Should remove empty and whitespace-only narratives
        assert all(filtered['Consumer complaint narrative'].str.strip().astype(bool))

    def test_chunk_text(self):
        long_text = " ".join(["word"] * 500)  # Create 500-word text
        chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) == 6  # Should be about 5 chunks with overlap
        
        # Verify overlap works
        last_part = chunks[0].split()[-20:]
        first_part = chunks[1].split()[:20]
        assert last_part == first_part
        
        # Test with short text (shouldn't chunk)
        short_text = "This is a short text"
        chunks = chunk_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_edge_cases(self):
        # Test empty string
        assert chunk_text("") == []
        
        # Test whitespace only
        assert chunk_text("   ") == []
        
        # Test exactly chunk size
        exact_text = " ".join(["word"] * 100)
        chunks = chunk_text(exact_text, chunk_size=100)
        assert len(chunks) == 1

    @pytest.mark.parametrize("input_text,expected", [
        ("Normal text", "normal text"),
        ("UPPER CASE", "upper case"),
        ("  trimmed  ", "trimmed"),
        ("special!@#chars", "special chars"),
        ("", ""),
        (None, ""),
        ("123 numbers 456", "123 numbers 456")
    ])
    def test_clean_text_variations(self, input_text, expected):
        assert clean_text(input_text) == expected