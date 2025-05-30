"""Utilities for validating and managing bibliographic references."""

import os
import re
import requests
import bibtexparser
from typing import Dict, List, Optional, Set, Tuple
import datetime

class ReferenceValidator:
    """Class for handling reference validation and processing."""
    
    def __init__(self, bib_path: str):
        """Initialize with path to BibTeX file.
        
        Args:
            bib_path: Path to the BibTeX file
        """
        self.bib_path = bib_path
        self.references = {}
        self.valid_refs = set()
        self.invalid_refs = set()
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
            
        try:
            headers = {"Accept": "application/json"}
            response = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def validate_references(self) -> Tuple[Set[str], Set[str]]:
        """Validate all references in the BibTeX file.
        
        Returns:
            Tuple of (valid_references, invalid_references) as sets of reference IDs
        """
        valid_refs = set()
        invalid_refs = set()
        
        for ref_id, entry in self.references.items():
            doi = entry.get('doi', '')
            if doi and self.validate_doi(doi):
                valid_refs.add(ref_id)
            else:
                invalid_refs.add(ref_id)
        
        self.valid_refs = valid_refs
        self.invalid_refs = invalid_refs
        return valid_refs, invalid_refs
    
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
