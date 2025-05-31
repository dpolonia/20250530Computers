"""
Reference service implementation.

This module implements the reference service interface, providing functionality for
validating and updating references.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set

from src.core.context import RevisionContext
from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.reference import Reference, ReferenceType
from src.services.interfaces import ReferenceServiceInterface
from src.adapters.bibtex_adapter import BibtexAdapter
from src.core.constants import SYSTEM_PROMPTS
from src.core.json_utils import parse_json_safely


class ReferenceService(ReferenceServiceInterface):
    """
    Service for reference management.
    
    This service is responsible for validating existing references and adding new
    ones based on reviewer suggestions.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the reference service.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.bibtex_adapter = BibtexAdapter(context)
    
    def validate_and_update_references(
        self,
        paper: Paper,
        reviewer_comments: List[ReviewerComment],
        output_path: Optional[str] = None
    ) -> List[Reference]:
        """
        Validate existing references and add new ones based on reviewer suggestions.
        
        Args:
            paper: Paper domain entity
            reviewer_comments: List of ReviewerComment domain entities
            output_path: Path where the new BibTeX file should be saved (optional)
            
        Returns:
            List of new Reference domain entities
        """
        self.logger.info("Validating and updating references")
        
        # Determine output path if not provided
        if not output_path:
            output_dir = self.context.output_dir or os.path.dirname(self.context.original_paper_path)
            output_path = os.path.join(output_dir, "updated_references.bib")
        
        # 1. Validate existing references
        valid_refs, invalid_refs = self._validate_references(paper.references)
        
        # Log validation results
        self.logger.info(f"Reference validation: {len(valid_refs)} valid, {len(invalid_refs)} invalid")
        for ref in invalid_refs:
            self.logger.warning(f"Invalid reference: {ref.id} - {ref.title}")
        
        # 2. Extract suggested references from reviewer comments
        suggested_refs = self._extract_suggested_references(reviewer_comments)
        
        # 3. Add new references
        new_refs = []
        for ref_data in suggested_refs:
            try:
                # Add reference to BibTeX
                ref_id = self.bibtex_adapter.add_reference(ref_data)
                
                # Create Reference domain entity
                ref_type = ReferenceType.ARTICLE  # Default type
                if "book" in ref_data.get("type", "").lower():
                    ref_type = ReferenceType.BOOK
                elif "conference" in ref_data.get("type", "").lower() or "proceedings" in ref_data.get("type", "").lower():
                    ref_type = ReferenceType.CONFERENCE
                
                new_ref = Reference(
                    id=ref_id,
                    title=ref_data.get("title", ""),
                    authors=ref_data.get("authors", []),
                    year=int(ref_data.get("year", 0)),
                    venue=ref_data.get("venue", ""),
                    reference_type=ref_type,
                    doi=ref_data.get("doi"),
                    url=ref_data.get("url"),
                    reason=ref_data.get("reason")
                )
                
                new_refs.append(new_ref)
                self.logger.info(f"Added new reference: {ref_id} - {new_ref.title}")
                
            except Exception as e:
                self.logger.error(f"Error adding reference: {e}")
        
        # 4. Export all references to BibTeX file
        all_refs = valid_refs + new_refs
        ref_ids = [ref.id for ref in all_refs]
        
        try:
            # Export references to BibTeX file
            bib_path = self.bibtex_adapter.export_references(ref_ids, output_path)
            self.logger.info(f"References exported to {bib_path}")
        except Exception as e:
            self.logger.error(f"Error exporting references: {e}")
        
        return new_refs
    
    def _validate_references(self, references: List[Reference]) -> Tuple[List[Reference], List[Reference]]:
        """
        Validate references.
        
        Args:
            references: List of Reference domain entities
            
        Returns:
            Tuple of (valid_references, invalid_references)
        """
        valid_refs = []
        invalid_refs = []
        
        # Extract reference IDs
        ref_ids = [ref.id for ref in references]
        
        # Validate references using adapter
        valid_ids, invalid_ids = self.bibtex_adapter.validate_references(ref_ids)
        
        # Map validation results back to Reference objects
        id_to_ref = {ref.id: ref for ref in references}
        
        for ref_id in valid_ids:
            if ref_id in id_to_ref:
                ref = id_to_ref[ref_id]
                ref.valid = True
                valid_refs.append(ref)
        
        for ref_id in invalid_ids:
            if ref_id in id_to_ref:
                ref = id_to_ref[ref_id]
                ref.valid = False
                invalid_refs.append(ref)
        
        return valid_refs, invalid_refs
    
    def _extract_suggested_references(self, reviewer_comments: List[ReviewerComment]) -> List[Dict[str, Any]]:
        """
        Extract suggested references from reviewer comments.
        
        Args:
            reviewer_comments: List of ReviewerComment domain entities
            
        Returns:
            List of reference data dictionaries
        """
        # Collect all reference-related comments
        ref_comments = []
        for reviewer in reviewer_comments:
            ref_comments.extend(reviewer.references_comments)
            
            # Also check for reference suggestions in main concerns and required changes
            for comment in reviewer.main_concerns + reviewer.required_changes + reviewer.suggested_changes:
                if any(term in comment.lower() for term in ["reference", "citation", "cite", "literature", "paper by", "work of", "study by"]):
                    ref_comments.append(comment)
            
            # Check detailed comments for reference suggestions
            for comment in reviewer.detailed_comments:
                if any(term in comment.text.lower() for term in ["reference", "citation", "cite", "literature", "paper by", "work of", "study by"]):
                    ref_comments.append(comment.text)
        
        if not ref_comments:
            self.logger.info("No reference suggestions found in reviewer comments")
            return []
        
        # Create prompt for reference extraction
        prompt = f"""
        I'm extracting suggested references from reviewer comments for a scientific paper. 
        Identify any references that reviewers suggest adding.
        
        REVIEWER COMMENTS ABOUT REFERENCES:
        {chr(10).join(ref_comments)}
        
        Based on these comments, extract all suggested references. For each reference, provide as much information as possible:
        1. Title of the paper/book
        2. Authors
        3. Year
        4. Venue (journal/conference)
        5. DOI or URL if mentioned
        6. The reason the reviewer suggested this reference
        
        IMPORTANT: Format the response as a valid JSON array of reference objects. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "title": "Deep Learning for Natural Language Processing",
            "authors": ["Smith, J.", "Johnson, A."],
            "year": 2020,
            "venue": "Journal of Machine Learning",
            "type": "article",
            "doi": "10.1234/jml.2020.1234",
            "reason": "Provides recent advances in NLP techniques relevant to the paper's methodology"
          }}
        ]
        
        If no specific references are suggested, return an empty array ([]).
        """
        
        # Get references from LLM
        references_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS.get("reference_extraction", "You are an expert in academic citations"),
            max_tokens=2000 if self.context.optimize_costs else 3000
        )
        
        # Parse the LLM response
        try:
            suggested_refs = parse_json_safely(references_json)
            return suggested_refs
        except Exception as e:
            self.logger.error(f"Error parsing suggested references: {e}")
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