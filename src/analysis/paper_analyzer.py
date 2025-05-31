"""
Paper analyzer module for extracting and analyzing paper content.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import os

from src.core.context import RevisionContext
from src.utils.pdf_processor import PDFProcessor
from src.core.json_utils import parse_json_safely
from src.core.constants import SYSTEM_PROMPTS
from src.analysis.interfaces import PaperAnalyzerInterface


class PaperAnalyzer(PaperAnalyzerInterface):
    """
    Analyzes the structure and content of academic papers.
    
    This class is responsible for extracting information from the original paper,
    including sections, tables, figures, and other elements. It uses LLM-based
    analysis to identify key components and their relationships.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the paper analyzer.
        
        Args:
            context: The shared revision context
        """
        super().__init__(context)
        self.logger = context.logger or logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform analysis and return structured results.
        
        This is an implementation of the abstract method from the Analyzer interface.
        
        Returns:
            Dictionary with analysis results
        """
        return self.analyze_paper()
    
    def analyze_paper(self) -> Dict[str, Any]:
        """
        Analyze the original paper and extract structured information.
        
        Returns:
            Dictionary with paper analysis results
        """
        self.logger.info("Analyzing original paper")
        
        # Process the PDF
        pdf_processor = PDFProcessor(self.context.original_paper_path)
        self.context.process_statistics["files_processed"] = self.context.process_statistics.get("files_processed", 0) + 1
        
        # Extract text and metadata
        text = pdf_processor.text
        metadata = pdf_processor.get_metadata()
        
        # Extract tables, figures, and sections
        tables = pdf_processor.extract_tables()
        figures = pdf_processor.extract_figures()
        sections = pdf_processor.extract_sections()
        
        # Use optimized token usage - focus on first part of the paper for analysis
        if self.context.optimize_costs:
            # Take the first 8000 chars of the paper for analysis
            analysis_text = text[:8000]
            self.logger.debug(f"Optimized token usage: Trimmed paper from {len(text)} to {len(analysis_text)} chars")
        else:
            # Use as much text as possible, but still limit to avoid token limit issues
            analysis_text = text[:20000]
            if len(text) > 20000:
                self.logger.debug(f"Trimmed paper from {len(text)} to {len(analysis_text)} chars to fit token limits")
        
        # Create prompt for paper analysis
        prompt = f"""
        I'm analyzing a scientific paper. Extract the key information and structure.
        
        PAPER TEXT (partial):
        ```
        {analysis_text}
        ```
        
        EXTRACTED METADATA:
        {metadata}
        
        Based on this information, provide a comprehensive analysis of the paper, including:
        1. Title and authors
        2. Abstract summary
        3. Main objectives of the paper
        4. Methodology used
        5. Key findings and results
        6. Limitations mentioned
        7. Conclusions
        
        IMPORTANT: Format the response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '{{' and end with '}}'.
        
        Example format:
        {{
          "title": "Paper title here",
          "authors": ["Author 1", "Author 2"],
          "abstract": "Brief summary of the abstract",
          "objectives": ["Main objective 1", "Main objective 2"],
          "methodology": "Description of methodology",
          "findings": ["Key finding 1", "Key finding 2"],
          "limitations": ["Limitation 1", "Limitation 2"],
          "conclusions": ["Conclusion 1", "Conclusion 2"]
        }}
        """
        
        # Use the LLM to analyze the paper
        paper_analysis_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["paper_analysis"],
            max_tokens=2000 if self.context.optimize_costs else 3000
        )
        
        try:
            # Parse the LLM response
            paper_analysis = parse_json_safely(paper_analysis_json)
            
            # Add extracted sections, tables, and figures
            paper_analysis["sections"] = sections
            paper_analysis["tables"] = tables
            paper_analysis["figures"] = [caption for caption, _ in figures]
            
            # Extract references from the PDF
            references = pdf_processor.extract_references()
            
            # Reduce token usage - only include a limited number of references
            if self.context.optimize_costs and len(references) > 10:
                # Keep only first 10 references to save tokens
                paper_analysis["references"] = references[:10]
                paper_analysis["references_count"] = len(references)
            else:
                paper_analysis["references"] = references
                
            # Add metadata
            paper_analysis["metadata"] = metadata
            
            self.logger.info("Paper analysis completed successfully")
            pdf_processor.close()
            return paper_analysis
            
        except ValueError as e:
            self.logger.error(f"Error parsing paper analysis: {e}")
            
            # Create a fallback analysis with basic information
            fallback_analysis = {
                "title": metadata.get("title", "Unknown Title"),
                "authors": metadata.get("author", "Unknown Authors").split(", "),
                "abstract": "Not available",
                "objectives": ["Not available"],
                "methodology": "Not available",
                "findings": ["Not available"],
                "limitations": ["Not available"],
                "conclusions": ["Not available"],
                "sections": sections,
                "tables": tables,
                "figures": [caption for caption, _ in figures],
                "references": references[:10] if len(references) > 10 else references,
                "metadata": metadata,
                "error": str(e)
            }
            
            self.logger.warning("Using fallback paper analysis due to parsing error")
            pdf_processor.close()
            return fallback_analysis
    
    def _get_completion(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """
        Get a completion from the LLM with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM response text
        """
        try:
            # Ensure LLM client is initialized
            if not self.context.llm_client:
                self.context.setup_llm_client()
                
            # Get completion from LLM
            response = self.context.llm_client.get_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
            
            # Update budget
            tokens_used = self.context.llm_client.total_tokens_used
            cost = self.context.llm_client.total_cost
            self.context.update_budget(tokens_used, cost)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting completion: {e}")
            raise