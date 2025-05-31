"""
BibTeX adapter implementation.

This module implements the BibTeX adapter interface, providing functionality for
interacting with BibTeX files.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set

from src.core.context import RevisionContext
from src.adapters.interfaces import BibtexAdapterInterface


class BibtexAdapter(BibtexAdapterInterface):
    """
    Adapter for BibTeX file operations.
    
    This adapter is responsible for reading, writing, and validating BibTeX files,
    providing functionality for reference management.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the BibTeX adapter.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.bibtex_entries = {}
        
        # Try to load initial BibTeX file if available
        if hasattr(context, 'bibtex_path') and context.bibtex_path:
            self.read(context.bibtex_path)
    
    def read(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a BibTeX file.
        
        Args:
            file_path: Path to the BibTeX file
            
        Returns:
            Dictionary of BibTeX entries
        """
        self.logger.info(f"Reading BibTeX file {file_path}")
        
        try:
            # Try using bibtexparser
            try:
                import bibtexparser
                from bibtexparser.bparser import BibTexParser
                
                with open(file_path, 'r', encoding='utf-8') as bibtex_file:
                    parser = BibTexParser(common_strings=True)
                    bib_database = bibtexparser.load(bibtex_file, parser)
                    
                # Convert to dictionary keyed by entry ID
                entries = {}
                for entry in bib_database.entries:
                    entry_id = entry.get('ID', '')
                    if entry_id:
                        entries[entry_id] = entry
                
                self.bibtex_entries.update(entries)
                return entries
                
            except ImportError:
                self.logger.warning("bibtexparser not installed. Using fallback method.")
                return self._fallback_read(file_path)
                
        except Exception as e:
            self.logger.error(f"Error reading BibTeX file: {e}")
            return {}
    
    def _fallback_read(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Fallback method to read a BibTeX file without bibtexparser.
        
        Args:
            file_path: Path to the BibTeX file
            
        Returns:
            Dictionary of BibTeX entries
        """
        entries = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as bibtex_file:
                content = bibtex_file.read()
            
            # Simple regex-based parsing
            entry_pattern = re.compile(r'@(\w+)\s*{\s*([^,]+),\s*(.*?)\s*}\s*', re.DOTALL)
            field_pattern = re.compile(r'\s*(\w+)\s*=\s*{(.*?)},?', re.DOTALL)
            
            for match in entry_pattern.finditer(content):
                entry_type = match.group(1)
                entry_id = match.group(2)
                fields_text = match.group(3)
                
                # Parse fields
                fields = {'ENTRYTYPE': entry_type, 'ID': entry_id}
                for field_match in field_pattern.finditer(fields_text):
                    field_name = field_match.group(1)
                    field_value = field_match.group(2)
                    fields[field_name] = field_value
                
                entries[entry_id] = fields
            
            self.bibtex_entries.update(entries)
            return entries
            
        except Exception as e:
            self.logger.error(f"Error in fallback BibTeX reading: {e}")
            return {}
    
    def write(self, content: Dict[str, Dict[str, Any]], file_path: str) -> str:
        """
        Write content to a BibTeX file.
        
        Args:
            content: Dictionary of BibTeX entries
            file_path: Path where the file should be saved
            
        Returns:
            Path to the written file
        """
        return self.export_references(list(content.keys()), file_path)
    
    def validate_references(self, references: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate references.
        
        Args:
            references: List of reference IDs to validate
            
        Returns:
            Tuple of (valid_references, invalid_references)
        """
        self.logger.info(f"Validating {len(references)} references")
        
        valid_refs = []
        invalid_refs = []
        
        for ref_id in references:
            # Check if reference exists in loaded entries
            if ref_id in self.bibtex_entries:
                entry = self.bibtex_entries[ref_id]
                
                # Basic validation checks
                valid = True
                
                # Check required fields based on entry type
                entry_type = entry.get('ENTRYTYPE', '').lower()
                
                if entry_type == 'article':
                    required_fields = ['author', 'title', 'journal', 'year']
                elif entry_type == 'book':
                    required_fields = ['author', 'title', 'publisher', 'year']
                elif entry_type in ['inproceedings', 'conference']:
                    required_fields = ['author', 'title', 'booktitle', 'year']
                else:
                    required_fields = ['author', 'title', 'year']
                
                for field in required_fields:
                    if field not in entry or not entry[field].strip():
                        valid = False
                        self.logger.warning(f"Reference {ref_id} missing required field: {field}")
                        break
                
                if valid:
                    valid_refs.append(ref_id)
                else:
                    invalid_refs.append(ref_id)
            else:
                # Reference not found in loaded entries
                invalid_refs.append(ref_id)
                self.logger.warning(f"Reference {ref_id} not found in loaded BibTeX entries")
        
        return valid_refs, invalid_refs
    
    def add_reference(self, reference_data: Dict[str, str]) -> str:
        """
        Add a new reference.
        
        Args:
            reference_data: Reference data
            
        Returns:
            Reference ID
        """
        self.logger.info("Adding new reference")
        
        # Generate a reference ID if not provided
        ref_id = reference_data.get('id', '')
        if not ref_id:
            # Generate ID from author and year
            authors = reference_data.get('authors', [])
            year = reference_data.get('year', '')
            
            if authors and year:
                # Extract last name of first author
                first_author = authors[0]
                last_name = first_author.split(',')[0] if ',' in first_author else first_author.split()[-1]
                
                # Clean up the last name
                last_name = re.sub(r'[^a-zA-Z]', '', last_name)
                
                # Create ID
                ref_id = f"{last_name.lower()}{year}"
                
                # Make sure it's unique
                base_id = ref_id
                counter = 1
                while ref_id in self.bibtex_entries:
                    ref_id = f"{base_id}{chr(96 + counter)}"  # add a, b, c, etc.
                    counter += 1
            else:
                # Fallback ID generation
                import uuid
                ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        
        # Determine entry type
        entry_type = reference_data.get('type', 'article').lower()
        if 'book' in entry_type:
            entry_type = 'book'
        elif 'conference' in entry_type or 'proceedings' in entry_type:
            entry_type = 'inproceedings'
        else:
            entry_type = 'article'
        
        # Create BibTeX entry
        entry = {'ID': ref_id, 'ENTRYTYPE': entry_type}
        
        # Map fields
        if 'title' in reference_data:
            entry['title'] = reference_data['title']
        
        if 'authors' in reference_data:
            authors = reference_data['authors']
            if isinstance(authors, list):
                entry['author'] = ' and '.join(authors)
            else:
                entry['author'] = authors
        
        if 'year' in reference_data:
            entry['year'] = str(reference_data['year'])
        
        if 'venue' in reference_data:
            venue = reference_data['venue']
            if entry_type == 'article':
                entry['journal'] = venue
            elif entry_type == 'inproceedings':
                entry['booktitle'] = venue
            else:
                entry['publisher'] = venue
        
        # Add optional fields
        for field in ['doi', 'url', 'pages', 'volume', 'number', 'publisher', 'abstract']:
            if field in reference_data and reference_data[field]:
                entry[field] = reference_data[field]
        
        # Add to BibTeX entries
        self.bibtex_entries[ref_id] = entry
        
        return ref_id
    
    def export_references(
        self,
        references: List[str],
        output_path: str
    ) -> str:
        """
        Export references to a BibTeX file.
        
        Args:
            references: List of reference IDs to export
            output_path: Path where the BibTeX file should be saved
            
        Returns:
            Path to the created BibTeX file
        """
        self.logger.info(f"Exporting {len(references)} references to {output_path}")
        
        try:
            # Try using bibtexparser
            try:
                import bibtexparser
                from bibtexparser.bwriter import BibTexWriter
                
                # Create database with selected entries
                entries = []
                for ref_id in references:
                    if ref_id in self.bibtex_entries:
                        entries.append(self.bibtex_entries[ref_id])
                
                db = bibtexparser.bibdatabase.BibDatabase()
                db.entries = entries
                
                # Write to file
                writer = BibTexWriter()
                with open(output_path, 'w', encoding='utf-8') as bibtex_file:
                    bibtex_file.write(writer.write(db))
                
                return output_path
                
            except ImportError:
                self.logger.warning("bibtexparser not installed. Using fallback method.")
                return self._fallback_export(references, output_path)
                
        except Exception as e:
            self.logger.error(f"Error exporting references: {e}")
            return ""
    
    def _fallback_export(
        self,
        references: List[str],
        output_path: str
    ) -> str:
        """
        Fallback method to export references without bibtexparser.
        
        Args:
            references: List of reference IDs to export
            output_path: Path where the BibTeX file should be saved
            
        Returns:
            Path to the created BibTeX file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as bibtex_file:
                for ref_id in references:
                    if ref_id in self.bibtex_entries:
                        entry = self.bibtex_entries[ref_id]
                        
                        # Write entry type and ID
                        entry_type = entry.get('ENTRYTYPE', 'misc')
                        bibtex_file.write(f"@{entry_type}{{{ref_id},\n")
                        
                        # Write fields
                        for key, value in entry.items():
                            if key not in ['ID', 'ENTRYTYPE']:
                                bibtex_file.write(f"  {key} = {{{value}}},\n")
                        
                        # Close entry
                        bibtex_file.write("}\n\n")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in fallback export: {e}")
            return ""