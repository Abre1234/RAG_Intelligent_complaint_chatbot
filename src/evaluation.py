# evaluation.py
import pandas as pd
from src.rag_pipeline import generate_answer
from typing import List, Dict
from datetime import datetime

class RAGEvaluator:
    def __init__(self):
        # Define your test questions and expected answers (if available)
        self.test_questions = [
            {
                "question": "What are the most common complaints about Credit Cards?",
                "product_filter": "Credit card",
                "expected_aspects": ["billing", "interest", "fees", "unauthorized charges"]
            },
            {
                "question": "Why are customers unhappy with Buy Now Pay Later services?",
                "product_filter": "Buy Now, Pay Later",
                "expected_aspects": ["late fees", "payment processing", "customer service"]
            },
            {
                "question": "What issues do customers report with money transfers?",
                "product_filter": "Money transfers",
                "expected_aspects": ["delays", "fees", "recipient issues"]
            },
            {
                "question": "How are customers complaining about savings accounts?",
                "product_filter": "Savings account",
                "expected_aspects": ["interest rates", "withdrawal limits", "fees"]
            },
            {
                "question": "What problems do customers face with personal loans?",
                "product_filter": "Personal loan",
                "expected_aspects": ["approval process", "repayment terms", "customer service"]
            }
        ]
    
    def evaluate_rag_system(self) -> pd.DataFrame:
        """Evaluate the RAG system on test questions and generate a report"""
        results = []
        
        for test_case in self.test_questions:
            question = test_case["question"]
            start_time = datetime.now()
            
            # Generate answer from RAG system
            answer, context = generate_answer(question)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze the quality of the answer
            quality_score = self._rate_answer_quality(
                answer, 
                test_case.get("expected_aspects", [])
            )
            
            # Prepare results for this question
            result = {
                "Question": question,
                "Generated Answer": answer,
                "Processing Time (sec)": f"{processing_time:.2f}",
                "Quality Score (1-5)": quality_score,
                "Retrieved Sources": self._format_context(context),
                "Analysis": self._generate_analysis(answer, context, test_case)
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _rate_answer_quality(self, answer: str, expected_aspects: List[str]) -> int:
        """Rate the answer quality from 1-5 based on coverage of expected aspects"""
        if not answer or "don't have enough information" in answer.lower():
            return 1
        
        answer_lower = answer.lower()
        matched_aspects = sum(1 for aspect in expected_aspects if aspect.lower() in answer_lower)
        
        # Score based on percentage of expected aspects covered
        coverage = matched_aspects / len(expected_aspects) if expected_aspects else 0
        
        if coverage > 0.8:
            return 5
        elif coverage > 0.6:
            return 4
        elif coverage > 0.4:
            return 3
        elif coverage > 0.2:
            return 2
        else:
            return 1
    
    def _format_context(self, context: List[str]) -> str:
        """Format the retrieved context chunks for display"""
        return "\n\n---\n\n".join(
            [f"Chunk {i+1}:\n{chunk[:200]}..." for i, chunk in enumerate(context)]
        )
    
    def _generate_analysis(self, answer: str, context: List[str], test_case: Dict) -> str:
        """Generate analysis of the answer quality"""
        analysis = []
        
        # Check if answer is relevant to the question
        if not any(word in answer.lower() for word in test_case["question"].lower().split()[:5]):
            analysis.append("Answer doesn't directly address the question.")
        
        # Check if answer uses the context
        if not any(chunk[:50] in answer for chunk in context):
            analysis.append("Answer doesn't appear to use the provided context effectively.")
        
        # Check for hallucinations
        if len(answer) > 500 and len(context) < 100:
            analysis.append("Potential hallucination - long answer with little context.")
        
        # Check for expected aspects
        missing_aspects = [
            aspect for aspect in test_case.get("expected_aspects", [])
            if aspect.lower() not in answer.lower()
        ]
        if missing_aspects:
            analysis.append(f"Missing expected aspects: {', '.join(missing_aspects)}")
        
        return " ".join(analysis) if analysis else "Answer meets all quality criteria."

def save_evaluation_report(df: pd.DataFrame, filename: str = "evaluation_report.md"):
    """Save the evaluation results to a markdown file"""
    with open(filename, "w") as f:
        f.write("# RAG System Evaluation Report\n\n")
        f.write("## Performance Overview\n\n")
        
        # Summary statistics
        avg_score = df["Quality Score (1-5)"].astype(float).mean()
        f.write(f"- **Average Quality Score**: {avg_score:.1f}/5\n")
        
        # Detailed results for each question
        for _, row in df.iterrows():
            f.write(f"\n## Question: {row['Question']}\n")
            f.write(f"- **Quality Score**: {row['Quality Score (1-5)']}/5\n")
            f.write(f"- **Processing Time**: {row['Processing Time (sec)']} seconds\n")
            f.write(f"\n### Generated Answer:\n{row['Generated Answer']}\n")
            f.write(f"\n### Retrieved Sources:\n{row['Retrieved Sources']}\n")
            f.write(f"\n### Analysis:\n{row['Analysis']}\n")
        
        f.write("\n## Conclusion\n")
        if avg_score >= 4:
            f.write("The RAG system is performing well, providing relevant and comprehensive answers.")
        elif avg_score >= 2.5:
            f.write("The RAG system shows promise but needs improvement in answer quality.")
        else:
            f.write("The RAG system needs significant improvements to be effective.")

if __name__ == "__main__":
    print("Running RAG system evaluation...")
    evaluator = RAGEvaluator()
    results_df = evaluator.evaluate_rag_system()
    
    # Display results in console
    pd.set_option('display.max_colwidth', 100)
    print("\nEvaluation Results:")
    print(results_df[["Question", "Quality Score (1-5)", "Processing Time (sec)", "Analysis"]])
    
    # Save detailed report
    save_evaluation_report(results_df)
    print("\nDetailed report saved to 'evaluation_report.md'")