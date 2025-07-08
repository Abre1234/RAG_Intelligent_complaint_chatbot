# tests/rag_test_pipeline.py
import pytest
from src.rag_pipeline import generate_answer
from src.retriever import ComplaintRetriever, retrieve_context
from unittest.mock import patch, MagicMock
import numpy as np

class TestRAGPipeline:
    @pytest.fixture
    def mock_retriever(self):
        with patch('src.retriever.ComplaintRetriever') as mock:
            mock_instance = mock.return_value
            mock_instance.search.return_value = [
                {
                    'text': 'Customers complain about high fees on BNPL services',
                    'product': 'Buy Now, Pay Later',
                    'score': 0.85,
                    'id': '12345'
                }
            ]
            yield mock_instance

    @pytest.fixture
    def mock_pipeline(self):
        with patch('src.rag_pipeline.qa_pipeline') as mock:
            mock.return_value = [{'generated_text': 'Customers are complaining about high fees and poor customer service in BNPL products.'}]
            yield mock

    def test_generate_answer_with_mocks(self, mock_retriever, mock_pipeline):
        question = "Why are customers complaining about BNPL?"
        answer, context = generate_answer(question)
        
        assert isinstance(answer, str)
        assert "complaining about high fees" in answer
        assert len(context) == 1
        assert "high fees on BNPL services" in context[0]
        
        # Verify the retriever was called with the question
        mock_retriever.search.assert_called_once_with(question, k=5)
        
        # Verify the pipeline was called
        mock_pipeline.assert_called_once()

    def test_retrieve_context_function(self):
        # Test the public retrieve_context interface
        with patch('src.retriever.ComplaintRetriever') as mock:
            mock_instance = mock.return_value
            mock_instance.search.return_value = [
                {'text': 'Test context 1', 'product': 'Credit card', 'score': 0.9, 'id': '111'},
                {'text': 'Test context 2', 'product': 'Credit card', 'score': 0.8, 'id': '112'}
            ]
            
            results = retrieve_context("test question", k=2)
            assert len(results) == 2
            assert "Test context 1" in results
            assert "Test context 2" in results

    def test_empty_context_handling(self):
        # Test how the system handles when no context is found
        with patch('src.retriever.ComplaintRetriever') as mock_retriever, \
             patch('src.rag_pipeline.qa_pipeline') as mock_pipeline:
            
            mock_retriever_instance = mock_retriever.return_value
            mock_retriever_instance.search.return_value = []
            
            mock_pipeline.return_value = [{'generated_text': "I don't have enough information to answer this question."}]
            
            answer, context = generate_answer("question with no results")
            
            assert "don't have enough information" in answer
            assert len(context) == 0

class TestRetrieverComponent:
    def test_retriever_initialization(self, tmp_path):
        # Create dummy index and metadata files for testing
        index_file = tmp_path / "complaints.index"
        metadata_file = tmp_path / "metadata.pkl"
        
        # Create a dummy FAISS index
        index = faiss.IndexFlatL2(384)  # Matching all-MiniLM-L6-v2 dimension
        faiss.write_index(index, str(index_file))
        
        # Create dummy metadata
        metadata = {
            'Product': ['Credit card', 'BNPL'],
            'cleaned_text': ['text 1', 'text 2'],
            'Complaint ID': ['1', '2'],
            'Issue': ['issue 1', 'issue 2']
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Test initialization
        retriever = ComplaintRetriever(
            index_path=str(index_file),
            metadata_path=str(metadata_file)
        
        assert retriever.index.ntotal == 2
        assert len(retriever.metadata) == 2