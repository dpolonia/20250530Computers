"""Journal Analyzer for analyzing and storing journal information.

This module provides utilities for retrieving and analyzing journal information
using the Scopus API, and storing it in the workflow database for use in the 
paper revision process.
"""

import os
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple

from .scopus_client import get_scopus_client
from .workflow_db import WorkflowDB

# Configure logging
logger = logging.getLogger(__name__)

class JournalAnalyzer:
    """Class for analyzing and storing journal information."""
    
    def __init__(self, db_path: str = "./.cache/workflow.db", scopus_api_key: str = None):
        """Initialize the journal analyzer.
        
        Args:
            db_path: Path to the SQLite database file
            scopus_api_key: Scopus API key (optional)
        """
        self.db = WorkflowDB(db_path)
        self.scopus_client = get_scopus_client(api_key=scopus_api_key)
    
    def analyze_journal_by_issn(self, issn: str) -> Dict[str, Any]:
        """Analyze a journal by ISSN and store information in the database.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Dictionary with journal information
        """
        # Check if we already have this journal in the database
        journal = self.db.get_journal_by_issn(issn)
        
        if journal:
            logger.info(f"Journal with ISSN {issn} already exists in database")
            return journal
        
        # Get journal information from Scopus
        journal_info = self.scopus_client.get_journal_info(issn)
        
        if not journal_info or journal_info.get('title') == "Unknown Journal":
            logger.warning(f"Could not find journal with ISSN {issn}")
            return {}
        
        # Store journal information in the database
        journal_id = self.db.store_journal(journal_info)
        
        # Store journal metrics
        if "metrics" in journal_info:
            metrics_data = journal_info["metrics"]
            metrics_data["year"] = datetime.datetime.now().year
            self.db.store_journal_metrics(journal_id, metrics_data)
        
        # Store top papers
        if "top_papers" in journal_info:
            for paper in journal_info["top_papers"]:
                paper_data = {
                    "scopus_id": paper.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
                    "doi": paper.get("prism:doi", ""),
                    "title": paper.get("dc:title", ""),
                    "authors": paper.get("dc:creator", ""),
                    "publication_year": paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in paper else 0,
                    "volume": paper.get("prism:volume", ""),
                    "issue": paper.get("prism:issueIdentifier", ""),
                    "pages": paper.get("prism:pageRange", ""),
                    "citations": int(paper.get("citedby-count", 0)),
                    "abstract": paper.get("dc:description", ""),
                    "keywords": "",
                    "ranking_type": "citations",
                    "ranking_value": int(paper.get("citedby-count", 0))
                }
                self.db.store_journal_top_paper(journal_id, paper_data)
        
        # Get reviewer preferences
        preferences = self.scopus_client.get_journal_reviewer_preferences(issn)
        
        for preference in preferences:
            self.db.store_journal_preference(journal_id, preference)
        
        logger.info(f"Successfully analyzed and stored journal with ISSN {issn}")
        
        # Return the journal summary
        return self.db.get_journal_summary(journal_id)
    
    def analyze_journal_by_title(self, title: str) -> Dict[str, Any]:
        """Analyze a journal by title and store information in the database.
        
        Args:
            title: Journal title
            
        Returns:
            Dictionary with journal information
        """
        # Check if we already have this journal in the database
        journal = self.db.get_journal_by_title(title)
        
        if journal:
            logger.info(f"Journal with title '{title}' already exists in database")
            return journal
        
        # Search for the journal ISSN using the title
        params = {
            "query": f"SRCTITLE(\"{title}\")",
            "count": 1,
            "field": "prism:issn"
        }
        
        try:
            response = self.scopus_client._make_request("/content/search/scopus", params)
            
            if "search-results" in response and "entry" in response["search-results"]:
                if len(response["search-results"]["entry"]) > 0:
                    journal_entry = response["search-results"]["entry"][0]
                    
                    if "prism:issn" in journal_entry:
                        issn = journal_entry["prism:issn"]
                        logger.info(f"Found ISSN {issn} for journal '{title}'")
                        return self.analyze_journal_by_issn(issn)
            
            logger.warning(f"Could not find ISSN for journal '{title}'")
            return {}
        except Exception as e:
            logger.error(f"Error searching for journal '{title}': {e}")
            return {}
    
    def find_similar_papers(self, run_id: str, journal_id: int, title: str, abstract: str, count: int = 5) -> List[Dict[str, Any]]:
        """Find papers in a journal that are similar to the given paper and store them in the database.
        
        Args:
            run_id: Run ID
            journal_id: Journal ID
            title: Paper title
            abstract: Paper abstract
            count: Maximum number of papers to return
            
        Returns:
            List of similar paper dictionaries
        """
        # Get journal information
        journal = self.db.get_journal_summary(journal_id)
        
        if not journal or "journal" not in journal:
            logger.warning(f"Journal with ID {journal_id} not found")
            return []
        
        journal_info = journal["journal"]
        
        # Check if we already have similar papers stored
        existing_papers = self.db.get_similar_journal_papers(run_id, journal_id)
        
        if existing_papers:
            logger.info(f"Found {len(existing_papers)} existing similar papers for run {run_id}")
            return existing_papers
        
        # Find similar papers using Scopus
        similar_papers = self.scopus_client.find_similar_papers_in_journal(
            title, abstract, 
            issn=journal_info.get("issn", ""), 
            journal_title=journal_info.get("title", ""),
            count=count
        )
        
        # Store the similar papers in the database
        stored_papers = []
        
        for paper in similar_papers:
            paper_data = {
                "scopus_id": paper.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
                "doi": paper.get("prism:doi", ""),
                "title": paper.get("dc:title", ""),
                "authors": paper.get("dc:creator", ""),
                "publication_year": paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in paper else 0,
                "volume": paper.get("prism:volume", ""),
                "issue": paper.get("prism:issueIdentifier", ""),
                "pages": paper.get("prism:pageRange", ""),
                "citations": int(paper.get("citedby-count", 0)),
                "abstract": paper.get("dc:description", ""),
                "keywords": "",
                "similarity_score": paper.get("similarity_score", 0.0),
                "similarity_reason": paper.get("similarity_reason", "")
            }
            
            paper_id = self.db.store_similar_journal_paper(run_id, journal_id, paper_data)
            stored_papers.append(paper_data)
        
        logger.info(f"Found and stored {len(stored_papers)} similar papers for run {run_id}")
        
        return stored_papers
    
    def get_journal_guidance(self, journal_id: int, manuscript_title: str, manuscript_abstract: str) -> Dict[str, Any]:
        """Generate guidance for submitting a manuscript to a journal.
        
        Args:
            journal_id: Journal ID
            manuscript_title: Title of the manuscript
            manuscript_abstract: Abstract of the manuscript
            
        Returns:
            Dictionary with guidance information
        """
        # Get journal information
        journal = self.db.get_journal_summary(journal_id)
        
        if not journal or "journal" not in journal:
            logger.warning(f"Journal with ID {journal_id} not found")
            return {}
        
        journal_info = journal["journal"]
        
        # Get similar papers
        similar_papers = self.scopus_client.find_similar_papers_in_journal(
            manuscript_title, manuscript_abstract,
            issn=journal_info.get("issn", ""),
            journal_title=journal_info.get("title", ""),
            count=5
        )
        
        # Generate guidance
        guidance = {
            "journal": journal_info,
            "metrics": journal.get("metrics", {}),
            "similar_papers": similar_papers,
            "suggestions": {
                "align_with_scope": True if similar_papers else False,
                "citation_impact": journal.get("metrics", {}).get("impact_factor", 0),
                "key_topics": [p.get("preference_value", "") for p in self.db.get_journal_preferences(journal_id) if p.get("preference_type") == "topic"][:5],
                "methodology_preferences": [p.get("preference_value", "") for p in self.db.get_journal_preferences(journal_id) if p.get("preference_type") == "methodology"][:5]
            }
        }
        
        return guidance
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def get_journal_analyzer(db_path: str = "./.cache/workflow.db", scopus_api_key: str = None) -> JournalAnalyzer:
    """Factory function to get a configured JournalAnalyzer.
    
    Args:
        db_path: Path to the SQLite database file
        scopus_api_key: Scopus API key (optional)
        
    Returns:
        Configured JournalAnalyzer instance
    """
    return JournalAnalyzer(db_path, scopus_api_key)


# Usage example (when run as script)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create an analyzer
    analyzer = get_journal_analyzer()
    
    # Analyze a journal
    journal = analyzer.analyze_journal_by_issn("0000-0000")  # Example ISSN
    
    # Find similar papers
    similar_papers = analyzer.find_similar_papers(
        "example_run_id", 
        journal.get("journal", {}).get("journal_id", 0),
        "Example Paper Title", 
        "Example paper abstract about a topic relevant to the journal."
    )
    
    # Close the connection
    analyzer.close()