"""Utilities for validating and managing bibliographic references.

This module provides functionality for validating and processing bibliographic
references, including DOI validation, BibTeX management, and integration with
the Scopus API for enhanced reference validation and retrieval.
"""

import os
import re
import requests
import bibtexparser
from typing import Dict, List, Optional, Set, Tuple, Any
import datetime
import logging

# Import the Scopus client
try:
    from .scopus_client import get_scopus_client, ScopusClient
    SCOPUS_AVAILABLE = True
except ImportError:
    SCOPUS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ReferenceValidator:
    """Class for handling reference validation and processing with Scopus integration."""
    
    def __init__(self, bib_path: str, use_scopus: bool = True, scopus_api_key: str = None):
        """Initialize with path to BibTeX file.
        
        Args:
            bib_path: Path to the BibTeX file
            use_scopus: Whether to use Scopus API for enhanced reference handling
            scopus_api_key: Scopus API key (optional)
        """
        self.bib_path = bib_path
        self.references = {}
        self.valid_refs = set()
        self.invalid_refs = set()
        self.use_scopus = use_scopus and SCOPUS_AVAILABLE
        self.scopus_client = None
        
        # Initialize Scopus client if available and requested
        if self.use_scopus:
            try:
                self.scopus_client = get_scopus_client(api_key=scopus_api_key)
                logger.info("Scopus client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Scopus client: {e}")
                self.use_scopus = False
        
        self._load_references()
    
    def _load_references(self):
        """Load references from the BibTeX file."""
        try:
            with open(self.bib_path, 'r', encoding='utf-8') as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)
                for entry in bib_database.entries:
                    self.references[entry.get('ID')] = entry
        except Exception as e:
            print(f"Error loading BibTeX file {self.bib_path}: {e}")
            raise
    
    def validate_doi(self, doi: str) -> bool:
        """Validate a DOI by checking if it resolves.
        
        Args:
            doi: DOI string to validate
            
        Returns:
            True if DOI is valid, False otherwise
        """
        if not doi:
            return False
            
        # Clean the DOI
        doi = doi.strip()
        if doi.lower().startswith('doi:'):
            doi = doi[4:].strip()
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:].strip()
        
        # Try Scopus validation first if available
        if self.use_scopus and self.scopus_client:
            try:
                paper = self.scopus_client.search_by_doi(doi)
                if paper:
                    return True
            except Exception as e:
                logger.debug(f"Scopus validation failed, falling back to DOI.org: {e}")
        
        # Fall back to DOI.org validation
        try:
            headers = {"Accept": "application/json"}
            response = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"DOI validation failed: {e}")
            return False
            
    def enhance_reference_with_scopus(self, ref_id: str) -> bool:
        """Enhance a reference with data from Scopus.
        
        Args:
            ref_id: Reference ID in the BibTeX database
            
        Returns:
            True if enhancement was successful, False otherwise
        """
        if not self.use_scopus or not self.scopus_client or ref_id not in self.references:
            return False
            
        entry = self.references[ref_id]
        doi = entry.get('doi', '')
        
        if not doi:
            # Try to find by title if DOI is not available
            title = entry.get('title', '')
            if title:
                papers = self.scopus_client.search_by_title(title, count=1)
                if papers:
                    paper = papers[0]
                    # Update reference with Scopus data
                    self._update_entry_from_scopus(entry, paper)
                    return True
            return False
        
        # Get paper data from Scopus
        paper = self.scopus_client.search_by_doi(doi)
        if paper:
            # Update reference with Scopus data
            self._update_entry_from_scopus(entry, paper)
            return True
            
        return False
        
    def _update_entry_from_scopus(self, entry: Dict[str, str], paper: Dict[str, Any]) -> None:
        """Update a BibTeX entry with data from Scopus.
        
        Args:
            entry: BibTeX entry to update
            paper: Scopus paper data
        """
        # Only update fields that are missing or empty
        if 'title' not in entry or not entry['title']:
            entry['title'] = paper.get('dc:title', '')
            
        if 'journal' not in entry or not entry['journal']:
            entry['journal'] = paper.get('prism:publicationName', '')
            
        if 'year' not in entry or not entry['year']:
            if 'prism:coverDate' in paper:
                entry['year'] = paper['prism:coverDate'].split('-')[0]
                
        if 'volume' not in entry or not entry['volume']:
            entry['volume'] = paper.get('prism:volume', '')
            
        if 'number' not in entry or not entry['number']:
            entry['number'] = paper.get('prism:issueIdentifier', '')
            
        if 'pages' not in entry or not entry['pages']:
            entry['pages'] = paper.get('prism:pageRange', '')
            
        if 'doi' not in entry or not entry['doi']:
            entry['doi'] = paper.get('prism:doi', '')
            
        if 'url' not in entry or not entry['url']:
            if 'prism:doi' in paper:
                entry['url'] = f"https://doi.org/{paper['prism:doi']}"
                
        if 'abstract' not in entry or not entry['abstract']:
            entry['abstract'] = paper.get('dc:description', '')
    
    def validate_references(self, fix_invalid: bool = True) -> Tuple[Set[str], Set[str]]:
        """Validate all references in the BibTeX file.
        
        Args:
            fix_invalid: Whether to attempt to fix invalid references using Scopus
            
        Returns:
            Tuple of (valid_references, invalid_references) as sets of reference IDs
        """
        valid_refs = set()
        invalid_refs = set()
        fixed_refs = set()
        
        for ref_id, entry in self.references.items():
            doi = entry.get('doi', '')
            if doi and self.validate_doi(doi):
                valid_refs.add(ref_id)
                
                # Optionally enhance valid references with additional Scopus data
                if self.use_scopus and self.scopus_client:
                    self.enhance_reference_with_scopus(ref_id)
            else:
                invalid_refs.add(ref_id)
        
        # Try to fix invalid references if requested
        if fix_invalid and self.use_scopus and self.scopus_client:
            for ref_id in list(invalid_refs):  # Use list to allow modification during iteration
                if self.fix_reference(ref_id):
                    invalid_refs.remove(ref_id)
                    valid_refs.add(ref_id)
                    fixed_refs.add(ref_id)
        
        self.valid_refs = valid_refs
        self.invalid_refs = invalid_refs
        
        if fixed_refs:
            logger.info(f"Fixed {len(fixed_refs)} invalid references using Scopus")
            
        return valid_refs, invalid_refs
        
    def fix_reference(self, ref_id: str) -> bool:
        """Attempt to fix an invalid reference using Scopus.
        
        Args:
            ref_id: Reference ID to fix
            
        Returns:
            True if the reference was fixed, False otherwise
        """
        if not self.use_scopus or not self.scopus_client or ref_id not in self.references:
            return False
            
        entry = self.references[ref_id]
        title = entry.get('title', '')
        authors = entry.get('author', '')
        
        if not title:
            return False
            
        # Try to find by title
        try:
            papers = self.scopus_client.search_by_title(title, count=3)
            
            # If we have multiple matches, try to filter by author
            if len(papers) > 1 and authors:
                author_last_name = authors.split(',')[0].split(' ')[0].lower()
                
                for paper in papers:
                    paper_authors = paper.get('dc:creator', '').lower()
                    if author_last_name in paper_authors:
                        # Update reference with Scopus data
                        self._update_entry_from_scopus(entry, paper)
                        
                        # Validate the updated DOI
                        if 'doi' in entry and self.validate_doi(entry['doi']):
                            return True
                        break
            
            # If no author match or only one paper, use the first result
            if papers:
                self._update_entry_from_scopus(entry, papers[0])
                
                # Validate the updated DOI
                if 'doi' in entry and self.validate_doi(entry['doi']):
                    return True
        except Exception as e:
            logger.debug(f"Error fixing reference {ref_id}: {e}")
            
        return False
    
    def export_references(self, ref_ids: Set[str], output_path: str) -> str:
        """Export selected references to a new BibTeX file.
        
        Args:
            ref_ids: Set of reference IDs to export
            output_path: Path for the output file
            
        Returns:
            Path to the created BibTeX file
        """
        # Create a new BibTeX database
        db = bibtexparser.bibdatabase.BibDatabase()
        db.entries = [self.references[ref_id] for ref_id in ref_ids if ref_id in self.references]
        
        # Write the database to file
        with open(output_path, 'w', encoding='utf-8') as bibtex_file:
            bibtexparser.dump(db, bibtex_file)
        
        return output_path
    
    def get_citation_style(self) -> str:
        """Analyze the BibTeX file to determine citation style.
        
        Returns:
            String describing the citation style
        """
        # This is a simplified approach - in a real implementation,
        # we would analyze more details of the citation format
        journal_names = []
        author_formats = []
        
        for entry in self.references.values():
            if 'journal' in entry:
                journal_names.append(entry['journal'])
            if 'author' in entry:
                author_formats.append(entry['author'])
        
        # Check for abbreviated journal names
        abbreviated = any('.' in journal for journal in journal_names)
        
        # Check for author format (First Last vs. Last, F.)
        last_first = any(',' in author for author in author_formats)
        
        if abbreviated and last_first:
            return "IEEE style (abbreviated journals, Last, F.I. author format)"
        elif not abbreviated and last_first:
            return "APA style (full journal names, Last, F.I. author format)"
        elif abbreviated and not last_first:
            return "Custom style with abbreviated journals"
        else:
            return "Author-year style with full journal names"
    
    def add_reference(self, entry: Dict[str, str]) -> str:
        """Add a new reference to the BibTeX database.
        
        Args:
            entry: Dictionary with BibTeX entry fields
            
        Returns:
            ID of the added reference
        """
        # Generate a reference ID if not provided
        if 'ID' not in entry:
            if 'author' in entry and 'year' in entry:
                # Create ID from first author's last name and year
                author = entry['author'].split(',')[0].split(' ')[0]
                entry['ID'] = f"{author}{entry['year']}"
            else:
                # Generate a unique ID
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                entry['ID'] = f"ref_{timestamp}"
        
        # Add to references dictionary
        self.references[entry['ID']] = entry
        
        return entry['ID']
    
    def save_references(self, output_path: Optional[str] = None):
        """Save all references to BibTeX file.
        
        Args:
            output_path: Path for saving the file. If None, original path is used.
        """
        save_path = output_path if output_path else self.bib_path
        
        # Create a new BibTeX database
        db = bibtexparser.bibdatabase.BibDatabase()
        db.entries = list(self.references.values())
        
        # Write the database to file
        with open(save_path, 'w', encoding='utf-8') as bibtex_file:
            bibtexparser.dump(db, bibtex_file)
    
    def find_similar_papers(self, ref_id: str, count: int = 5) -> List[Dict[str, str]]:
        """Find papers similar to the given reference using Scopus.
        
        Args:
            ref_id: Reference ID to find similar papers for
            count: Maximum number of similar papers to return
            
        Returns:
            List of similar papers in BibTeX entry format
        """
        if not self.use_scopus or not self.scopus_client or ref_id not in self.references:
            return []
            
        entry = self.references[ref_id]
        title = entry.get('title', '')
        abstract = entry.get('abstract', '')
        
        if not title:
            return []
        
        try:
            similar_papers = self.scopus_client.recommend_similar_papers(title, abstract, count=count)
            
            # Convert to BibTeX format
            return [self.scopus_client.format_bibtex_entry(paper) for paper in similar_papers]
        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
            return []
    
    def get_citation_report(self, ref_id: str) -> Optional[Dict[str, Any]]:
        """Generate a citation report for a reference using Scopus.
        
        Args:
            ref_id: Reference ID to get citation report for
            
        Returns:
            Citation report dictionary or None if unavailable
        """
        if not self.use_scopus or not self.scopus_client or ref_id not in self.references:
            return None
            
        entry = self.references[ref_id]
        doi = entry.get('doi', '')
        
        if not doi:
            return None
            
        try:
            return self.scopus_client.generate_citation_report(doi)
        except Exception as e:
            logger.error(f"Error generating citation report: {e}")
            return None
    
    def add_reference_from_doi(self, doi: str) -> Optional[str]:
        """Add a new reference by DOI using Scopus.
        
        Args:
            doi: DOI of the paper to add
            
        Returns:
            Reference ID if added successfully, None otherwise
        """
        if not self.use_scopus or not self.scopus_client:
            return None
            
        # Clean the DOI
        doi = doi.strip()
        if doi.lower().startswith('doi:'):
            doi = doi[4:].strip()
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:].strip()
            
        try:
            paper = self.scopus_client.search_by_doi(doi)
            if not paper:
                return None
                
            bibtex_entry = self.scopus_client.format_bibtex_entry(paper)
            ref_id = self.add_reference(bibtex_entry)
            return ref_id
        except Exception as e:
            logger.error(f"Error adding reference from DOI: {e}")
            return None
    
    def search_and_add_references(self, query: str, count: int = 5) -> List[str]:
        """Search for papers and add them as references.
        
        Args:
            query: Search query
            count: Maximum number of references to add
            
        Returns:
            List of added reference IDs
        """
        if not self.use_scopus or not self.scopus_client:
            return []
            
        try:
            # Determine if the query is a title search or author search
            if "author:" in query.lower():
                author = query.lower().split("author:")[1].strip()
                papers = self.scopus_client.search_by_author(author, count=count)
            else:
                papers = self.scopus_client.search_by_title(query, count=count)
                
            added_refs = []
            for paper in papers:
                bibtex_entry = self.scopus_client.format_bibtex_entry(paper)
                ref_id = self.add_reference(bibtex_entry)
                added_refs.append(ref_id)
                
            return added_refs
        except Exception as e:
            logger.error(f"Error searching and adding references: {e}")
            return []
