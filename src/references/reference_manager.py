"""
Reference manager module for validating and updating paper references.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Set

from src.core.context import RevisionContext
from src.utils.reference_validator import ReferenceValidator
from src.core.json_utils import parse_json_safely
from src.core.constants import SYSTEM_PROMPTS


class ReferenceManager:
    """
    Validates and updates references in academic papers.
    
    This class is responsible for validating existing references, suggesting new
    references based on reviewer comments, and updating the bibliography file.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the reference manager.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def validate_and_update_references(
        self,
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Validate existing references and add new ones based on reviewer suggestions.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            output_path: Path where the new BibTeX file should be saved (optional)
            
        Returns:
            List of new references added
        """
        self.logger.info("Validating and updating references")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = self.context.get_output_path("references.bib")
        
        # Load references
        # Only use Scopus if API integration is enabled
        use_scopus = hasattr(self.context, 'api') and self.context.api is not None and "scopus" in self.context.api
        ref_validator = ReferenceValidator(self.context.bib_path, 
                                          use_scopus=use_scopus, 
                                          scopus_api_key=self.context.api_key if hasattr(self.context, 'api_key') else None)
        
        # Validate existing references
        valid_refs, invalid_refs = ref_validator.validate_references()
        self.logger.info(f"Found {len(valid_refs)} valid and {len(invalid_refs)} invalid references")
        
        # Extract reference suggestions from reviewer comments
        reference_comments = []
        for reviewer in reviewer_comments:
            ref_comments = reviewer.get("references_comments", [])
            reference_comments.extend(ref_comments)
        
        # Generate new references based on reviewer suggestions
        if reference_comments:
            prompt = f"""
            I'm updating references for a scientific paper based on reviewer comments.
            
            Current Paper References:
            {paper_analysis.get('references', ['No references available'])[:10]}
            
            Reviewer Comments on References:
            {reference_comments}
            
            Based on these comments, suggest new references that should be added to the paper.
            For each reference, provide:
            1. Title
            2. Authors
            3. Journal/Conference
            4. Year
            5. DOI (if you can estimate it)
            6. Why this reference should be added
            
            IMPORTANT: Format the response as a valid JSON array of objects with the fields above. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
            
            Example format:
            [
              {{
                "title": "Example Paper Title",
                "authors": "Author1, Author2",
                "journal": "Journal of Examples",
                "year": 2023,
                "doi": "10.1234/example.5678",
                "why": "Provides relevant methodology"
              }},
              ...
            ]
            """
            
            # Use the LLM to suggest new references
            new_refs_json = self._get_completion(
                prompt=prompt,
                system_prompt="You are a scientific reference assistant. Suggest new references based on reviewer comments.",
                max_tokens=2000 if self.context.optimize_costs else 3000
            )
            
            try:
                # Parse the LLM response
                new_refs_data = parse_json_safely(new_refs_json)
                new_references = []
                
                for ref_data in new_refs_data:
                    # Convert to BibTeX entry format
                    entry = {
                        "ENTRYTYPE": "article",
                        "title": ref_data.get("title", "Unknown Title"),
                        "author": ref_data.get("authors", "Unknown Authors"),
                        "journal": ref_data.get("journal", ref_data.get("conference", "Unknown Venue")),
                        "year": str(ref_data.get("year", "2023")),
                        "doi": ref_data.get("doi", "")
                    }
                    
                    # Add to reference validator
                    ref_id = ref_validator.add_reference(entry)
                    
                    new_references.append({
                        "id": ref_id,
                        "title": entry["title"],
                        "authors": entry["author"],
                        "year": entry["year"],
                        "reason": ref_data.get("why", "Suggested by reviewer")
                    })
                
                # Save updated references
                ref_validator.export_references(
                    set(list(valid_refs) + [ref["id"] for ref in new_references]),
                    output_path
                )
                self.context.process_statistics["files_created"] = self.context.process_statistics.get("files_created", 0) + 1
                
                self.logger.info(f"Added {len(new_references)} new references")
                return new_references
                
            except ValueError as e:
                self.logger.warning(f"LLM didn't return valid JSON for new references. Error: {e}. No new references added.")
        
        # If no new references, just export valid ones
        ref_validator.export_references(valid_refs, output_path)
        self.context.process_statistics["files_created"] = self.context.process_statistics.get("files_created", 0) + 1
        
        self.logger.info("No new references added")
        return []
    
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