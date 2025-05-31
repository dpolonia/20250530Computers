"""
Paper service implementation.

This module implements the paper service interface, providing functionality for
analyzing papers and extracting structured information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext
from src.domain.paper import Paper, Section, Table, Figure, Reference
from src.services.interfaces import PaperServiceInterface
from src.core.constants import SYSTEM_PROMPTS
from src.core.json_utils import parse_json_safely
from src.adapters.pdf_adapter import PDFAdapter


class PaperService(PaperServiceInterface):
    """
    Service for paper analysis and information extraction.
    
    This service is responsible for analyzing papers and extracting structured
    information about their content, structure, and metadata.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the paper service.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.paper = None
        self.pdf_adapter = PDFAdapter(context)
    
    def analyze_paper(self) -> Paper:
        """
        Analyze the paper and extract structured information.
        
        Returns:
            Paper domain entity
        """
        self.logger.info("Analyzing original paper")
        
        # Extract text and metadata from the PDF
        text = self.pdf_adapter.extract_text(self.context.original_paper_path)
        metadata = self.pdf_adapter.extract_metadata(self.context.original_paper_path)
        
        # Extract tables, figures, and sections
        tables_data = self.pdf_adapter.extract_tables(self.context.original_paper_path)
        figures_data = self.pdf_adapter.extract_figures(self.context.original_paper_path)
        sections_data = self.pdf_adapter.extract_sections(self.context.original_paper_path)
        
        # Create domain entities from extracted data
        sections = [Section(name=name, content=content) for name, content in sections_data.items()]
        tables = [Table(caption=table["caption"], content=[], number=i+1) 
                 for i, table in enumerate(tables_data)]
        figures = [Figure(caption=caption, path=path, number=i+1) 
                  for i, (caption, path) in enumerate(figures_data)]
        
        # Optimize token usage - use a smaller chunk of text for analysis
        if self.context.optimize_costs:
            analysis_text = text[:8000]
        else:
            analysis_text = text[:20000]
        
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
        
        # Get analysis from LLM
        paper_analysis_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["paper_analysis"],
            max_tokens=2000 if self.context.optimize_costs else 3000
        )
        
        try:
            # Parse the LLM response
            paper_data = parse_json_safely(paper_analysis_json)
            
            # Extract references
            refs = self.pdf_adapter.extract_references(self.context.original_paper_path)
            references = []
            for i, ref_text in enumerate(refs):
                # Parse reference text (simplified for now)
                parts = ref_text.split('. ', 1)
                if len(parts) > 1:
                    authors_year, title_rest = parts
                    authors = authors_year.split('(')[0].strip()
                    year_str = authors_year.split('(')[1].split(')')[0].strip() if '(' in authors_year else "2000"
                    try:
                        year = int(year_str)
                    except ValueError:
                        year = 2000
                    
                    venue = title_rest.split('. ', 1)[1] if '. ' in title_rest else "Unknown Venue"
                    title = title_rest.split('. ', 1)[0] if '. ' in title_rest else title_rest
                    
                    references.append(Reference(
                        key=f"ref{i+1}",
                        title=title,
                        authors=[authors],
                        year=year,
                        venue=venue
                    ))
            
            # Create the paper domain entity
            self.paper = Paper(
                title=paper_data.get("title", metadata.get("title", "Unknown Title")),
                authors=paper_data.get("authors", [metadata.get("author", "Unknown Author")]),
                abstract=paper_data.get("abstract", ""),
                sections=sections,
                tables=tables,
                figures=figures,
                references=references,
                objectives=paper_data.get("objectives", []),
                methodology=paper_data.get("methodology", ""),
                findings=paper_data.get("findings", []),
                limitations=paper_data.get("limitations", []),
                conclusions=paper_data.get("conclusions", [])
            )
            
            self.logger.info("Paper analysis completed successfully")
            return self.paper
            
        except ValueError as e:
            self.logger.error(f"Error parsing paper analysis: {e}")
            
            # Create a fallback paper entity with basic information
            self.paper = Paper(
                title=metadata.get("title", "Unknown Title"),
                authors=[metadata.get("author", "Unknown Author")],
                abstract="",
                sections=sections,
                tables=tables,
                figures=figures,
                references=references
            )
            
            self.logger.warning("Using fallback paper entity due to parsing error")
            return self.paper
    
    def get_section_by_name(self, name: str) -> Optional[str]:
        """
        Get a section by name.
        
        Args:
            name: The name of the section to find
            
        Returns:
            The section content if found, None otherwise
        """
        if not self.paper:
            self.analyze_paper()
        
        section = self.paper.get_section_by_name(name)
        return section.content if section else None
    
    def get_paper_metadata(self) -> Dict[str, Any]:
        """
        Get paper metadata.
        
        Returns:
            Dictionary with paper metadata
        """
        if not self.paper:
            self.analyze_paper()
        
        return {
            "title": self.paper.title,
            "authors": self.paper.authors,
            "abstract": self.paper.abstract,
            "section_count": self.paper.section_count,
            "reference_count": self.paper.reference_count,
            "word_count": self.paper.word_count
        }
    
    def _get_completion(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """
        Get a completion from the LLM with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If there was an error getting the completion
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