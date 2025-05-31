"""Citation Analyzer for cross-database academic citation analysis.

This module provides utilities for analyzing citation data across multiple academic
databases (Scopus and Web of Science) to provide a comprehensive view of a paper's
impact and to find the most relevant citations for a manuscript.
"""

import os
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .scopus_client import get_scopus_client
from .wos_client import get_wos_client
from .workflow_db import WorkflowDB

# Configure logging
logger = logging.getLogger(__name__)

class CitationAnalyzer:
    """Class for analyzing citation data across multiple academic databases."""
    
    def __init__(self, 
                 db_path: str = "./.cache/workflow.db", 
                 scopus_api_key: str = None,
                 wos_client_id: str = None,
                 wos_client_secret: str = None):
        """Initialize the citation analyzer.
        
        Args:
            db_path: Path to the SQLite database file
            scopus_api_key: Scopus API key (optional)
            wos_client_id: Web of Science API client ID (optional)
            wos_client_secret: Web of Science API client secret (optional)
        """
        self.db = WorkflowDB(db_path)
        self.scopus_client = get_scopus_client(api_key=scopus_api_key)
        
        # Try to initialize Web of Science client if credentials can be found
        self.wos_client = None
        self.wos_available = False
        
        # Check for WoS credentials
        if not wos_client_secret:
            # Try to get from environment or prompt user
            wos_client_secret = os.environ.get("WOS_CLIENT_SECRET", "")
            
            # If still not available, prompt user
            if not wos_client_secret:
                try:
                    from getpass import getpass
                    print("\nWeb of Science integration is available but requires client credentials.")
                    print("Do you have a Web of Science client secret? (yes/no): ", end="")
                    response = input().strip().lower()
                    
                    if response.startswith('y'):
                        wos_client_secret = getpass("Enter your Web of Science client secret: ")
                except Exception:
                    # In case of any issues with input (e.g., in non-interactive environments)
                    pass
        
        # Try to initialize WoS client if we have credentials
        try:
            if wos_client_id and wos_client_secret:
                # Import here to avoid errors if package is not installed
                try:
                    from ..utils.wos_client import get_wos_client
                    self.wos_client = get_wos_client(
                        client_id=wos_client_id, 
                        client_secret=wos_client_secret
                    )
                    self.wos_available = True
                    logger.info("Web of Science API client successfully initialized")
                    
                    # Save credentials to environment for future runs if they worked
                    os.environ["WOS_CLIENT_SECRET"] = wos_client_secret
                except ImportError:
                    logger.warning("Web of Science client package not installed. Run 'pip install wos' to enable WoS integration.")
            else:
                logger.info("Web of Science credentials not available. Using Scopus API only.")
        except Exception as e:
            logger.warning(f"Error initializing Web of Science client: {e}")
            self.wos_available = False
            self.wos_client = None
    
    def analyze_paper_citations(self, doi: str) -> Dict[str, Any]:
        """Analyze citation data for a paper across all available databases.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dictionary with citation analysis results
        """
        results = {
            "doi": doi,
            "analysis_date": datetime.datetime.now().isoformat(),
            "sources": ["scopus"],
            "citation_counts": {},
            "citation_history": [],
            "top_citing_journals": [],
            "top_citing_countries": [],
            "citation_network": {}
        }
        
        # Get Scopus citation data
        scopus_data = self._get_scopus_citation_data(doi)
        
        if scopus_data:
            results["citation_counts"]["scopus"] = scopus_data.get("total_citations", 0)
            results["citation_history"] = scopus_data.get("citation_history", [])
            results["top_citing_journals"] = scopus_data.get("top_citing_journals", [])
            results["top_citing_countries"] = scopus_data.get("top_citing_countries", [])
            
            # Get paper metadata
            scopus_paper = self.scopus_client.search_by_doi(doi)
            if scopus_paper:
                results["title"] = scopus_paper.get("dc:title", "")
                results["authors"] = scopus_paper.get("dc:creator", "")
                results["journal"] = scopus_paper.get("prism:publicationName", "")
                results["year"] = scopus_paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in scopus_paper else ""
        
        # Get Web of Science citation data if available
        if self.wos_available and self.wos_client:
            wos_data = self._get_wos_citation_data(doi)
            
            if wos_data:
                results["sources"].append("wos")
                results["citation_counts"]["wos"] = wos_data.get("citation_count", 0)
                
                # Add any additional data from WoS
                if "citing_journals" in wos_data:
                    # Merge with Scopus data
                    self._merge_journal_citations(results["top_citing_journals"], wos_data["citing_journals"])
        
        # Calculate combined metrics
        self._calculate_combined_metrics(results)
        
        return results
    
    def _get_scopus_citation_data(self, doi: str) -> Dict[str, Any]:
        """Get citation data from Scopus.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dictionary with Scopus citation data
        """
        try:
            # Get citation overview
            citation_report = self.scopus_client.generate_citation_report(doi)
            
            if citation_report and "citations" in citation_report:
                return citation_report["citations"]
            else:
                logger.warning(f"No Scopus citation data found for DOI: {doi}")
                return {}
        except Exception as e:
            logger.error(f"Error getting Scopus citation data: {e}")
            return {}
    
    def _get_wos_citation_data(self, doi: str) -> Dict[str, Any]:
        """Get citation data from Web of Science.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dictionary with Web of Science citation data
        """
        try:
            # Get paper details from WoS
            paper = self.wos_client.search_by_doi(doi)
            
            if not paper:
                logger.warning(f"Paper not found in Web of Science: {doi}")
                return {}
                
            # Extract WoS ID
            wos_id = paper.get("UID", "")
            
            if not wos_id:
                logger.warning(f"No WoS ID found for paper: {doi}")
                return {}
                
            # Get citing papers
            citing_papers = self.wos_client.get_citing_papers(wos_id)
            
            # Prepare result
            result = {
                "citation_count": len(citing_papers),
                "citing_papers": citing_papers
            }
            
            # Extract citing journals
            citing_journals = defaultdict(int)
            for paper in citing_papers:
                journal = paper.get("SourceTitle", "")
                if journal:
                    citing_journals[journal] += 1
            
            # Convert to list format similar to Scopus
            result["citing_journals"] = [
                {"journal": journal, "count": count}
                for journal, count in sorted(citing_journals.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return result
        except Exception as e:
            logger.error(f"Error getting Web of Science citation data: {e}")
            return {}
    
    def _merge_journal_citations(self, scopus_journals: List[Dict[str, Any]], wos_journals: List[Dict[str, Any]]):
        """Merge journal citation data from Scopus and Web of Science.
        
        Args:
            scopus_journals: List of journals from Scopus
            wos_journals: List of journals from Web of Science
        """
        # Create a dictionary of Scopus journals
        journal_dict = {j["journal"].lower(): j for j in scopus_journals}
        
        # Merge WoS journals
        for wos_journal in wos_journals:
            journal_name = wos_journal["journal"].lower()
            
            if journal_name in journal_dict:
                # Journal exists in Scopus, add WoS count
                journal_dict[journal_name]["wos_count"] = wos_journal["count"]
                # Update total (assumes Scopus count is more complete, but add any additional from WoS)
                if journal_dict[journal_name]["count"] < wos_journal["count"]:
                    journal_dict[journal_name]["count"] = wos_journal["count"]
            else:
                # New journal from WoS
                scopus_journals.append({
                    "journal": wos_journal["journal"],
                    "count": wos_journal["count"],
                    "wos_count": wos_journal["count"],
                    "source": "wos"
                })
        
        # Re-sort the combined list
        scopus_journals.sort(key=lambda x: x["count"], reverse=True)
    
    def _calculate_combined_metrics(self, results: Dict[str, Any]):
        """Calculate combined citation metrics across databases.
        
        Args:
            results: Dictionary with citation analysis results
        """
        # Calculate total citations across all sources
        total_citations = sum(results["citation_counts"].values())
        results["total_citations"] = total_citations
        
        # Calculate citation velocity (citations per year)
        if results.get("year") and results["year"].isdigit():
            years_since_publication = max(1, datetime.datetime.now().year - int(results["year"]))
            results["citation_velocity"] = total_citations / years_since_publication
        else:
            results["citation_velocity"] = 0
        
        # Add normalized metrics
        if "top_citing_journals" in results and results["top_citing_journals"]:
            # Calculate h-index for citing journals
            h_index = 0
            for i, journal in enumerate(results["top_citing_journals"], 1):
                if journal["count"] >= i:
                    h_index = i
                else:
                    break
            results["citing_journals_h_index"] = h_index
    
    def find_cross_database_papers(self, title: str, abstract: str, count: int = 10) -> List[Dict[str, Any]]:
        """Find papers across multiple databases based on title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            count: Maximum number of papers to return
            
        Returns:
            List of paper dictionaries with source information
        """
        all_papers = []
        
        # Get papers from Scopus
        scopus_papers = self.scopus_client.recommend_similar_papers(title, abstract, count=count)
        
        for paper in scopus_papers:
            # Format the paper data
            paper_data = {
                "title": paper.get("dc:title", ""),
                "authors": paper.get("dc:creator", ""),
                "journal": paper.get("prism:publicationName", ""),
                "year": paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in paper else "",
                "doi": paper.get("prism:doi", ""),
                "citations": int(paper.get("citedby-count", 0)),
                "abstract": paper.get("dc:description", ""),
                "source": "scopus",
                "score": paper.get("similarity_score", 1.0) if "similarity_score" in paper else 1.0
            }
            all_papers.append(paper_data)
        
        # Get papers from Web of Science if available
        if self.wos_available and self.wos_client:
            wos_papers = self.wos_client.find_similar_papers(title, abstract, count=count)
            
            for paper in wos_papers:
                # Check if paper already exists from Scopus (by DOI)
                doi = paper.get("DOI", "")
                if doi and any(p["doi"] == doi for p in all_papers):
                    continue
                    
                # Format the paper data
                paper_data = {
                    "title": paper.get("Title", ""),
                    "authors": ", ".join(paper.get("Authors", [])),
                    "journal": paper.get("SourceTitle", ""),
                    "year": paper.get("Year", ""),
                    "doi": doi,
                    "citations": int(paper.get("CitationCount", 0)),
                    "abstract": paper.get("Abstract", ""),
                    "source": "wos",
                    "score": 0.9  # Default score for WoS papers
                }
                all_papers.append(paper_data)
        
        # Sort combined results by score and citations
        all_papers.sort(key=lambda x: (x["score"], x["citations"]), reverse=True)
        
        # Limit to requested count
        return all_papers[:count]
    
    def get_citation_overlap(self, dois: List[str]) -> Dict[str, Any]:
        """Analyze citation overlap between multiple papers.
        
        Args:
            dois: List of DOIs to analyze
            
        Returns:
            Dictionary with citation overlap analysis
        """
        papers = []
        citing_papers = {}
        
        # Get data for each paper
        for doi in dois:
            # Get paper data from Scopus
            paper = self.scopus_client.search_by_doi(doi)
            
            if not paper:
                logger.warning(f"Paper not found in Scopus: {doi}")
                continue
                
            # Format paper data
            paper_data = {
                "doi": doi,
                "title": paper.get("dc:title", ""),
                "authors": paper.get("dc:creator", ""),
                "year": paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in paper else "",
                "citation_count": int(paper.get("citedby-count", 0))
            }
            papers.append(paper_data)
            
            # Get citing papers
            citations = self.scopus_client.get_citations(doi, count=100)
            citing_papers[doi] = [
                citation.get("dc:identifier", "").replace("SCOPUS_ID:", "")
                for citation in citations
            ]
        
        # Calculate overlap
        overlap_matrix = {}
        for i, doi1 in enumerate(dois):
            overlap_matrix[doi1] = {}
            for j, doi2 in enumerate(dois):
                if i == j:
                    overlap_matrix[doi1][doi2] = 1.0
                    continue
                    
                # Get common citing papers
                citing1 = set(citing_papers.get(doi1, []))
                citing2 = set(citing_papers.get(doi2, []))
                
                if not citing1 or not citing2:
                    overlap_matrix[doi1][doi2] = 0.0
                    continue
                    
                common = citing1.intersection(citing2)
                union = citing1.union(citing2)
                
                # Calculate Jaccard index
                overlap_matrix[doi1][doi2] = len(common) / len(union) if union else 0.0
        
        # Prepare result
        result = {
            "papers": papers,
            "overlap_matrix": overlap_matrix,
            "total_unique_citations": len(set().union(*[set(citing_papers.get(doi, [])) for doi in dois])),
            "common_citations": {}
        }
        
        # Find citations common to all papers
        if len(dois) > 1:
            all_citing_sets = [set(citing_papers.get(doi, [])) for doi in dois]
            common_to_all = set.intersection(*all_citing_sets)
            result["common_citations"]["common_to_all"] = len(common_to_all)
            
            # Find citations common to at least two papers
            common_to_some = set()
            for i in range(len(dois)):
                for j in range(i+1, len(dois)):
                    common = set(citing_papers.get(dois[i], [])).intersection(set(citing_papers.get(dois[j], [])))
                    common_to_some = common_to_some.union(common)
            
            result["common_citations"]["common_to_some"] = len(common_to_some)
        
        return result
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the citation analyzer.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "scopus_stats": self.scopus_client.get_usage_statistics()
        }
        
        if self.wos_available and self.wos_client:
            stats["wos_stats"] = self.wos_client.get_usage_statistics()
        
        return stats
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def get_citation_analyzer(db_path: str = "./.cache/workflow.db", 
                          scopus_api_key: str = None,
                          wos_client_id: str = None,
                          wos_client_secret: str = None) -> CitationAnalyzer:
    """Factory function to get a configured CitationAnalyzer.
    
    Args:
        db_path: Path to the SQLite database file
        scopus_api_key: Scopus API key (optional)
        wos_client_id: Web of Science API client ID (optional)
        wos_client_secret: Web of Science API client secret (optional)
        
    Returns:
        Configured CitationAnalyzer instance
    """
    return CitationAnalyzer(db_path, scopus_api_key, wos_client_id, wos_client_secret)


# Usage example (when run as script)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create an analyzer
    analyzer = get_citation_analyzer()
    
    # Analyze a paper
    citation_data = analyzer.analyze_paper_citations("10.1016/j.cose.2022.102644")
    
    # Print results
    print(f"Citation analysis for DOI: {citation_data.get('doi')}")
    print(f"Title: {citation_data.get('title')}")
    print(f"Sources: {', '.join(citation_data.get('sources', []))}")
    print(f"Total citations: {citation_data.get('total_citations', 0)}")
    
    for source, count in citation_data.get('citation_counts', {}).items():
        print(f"  {source.upper()}: {count}")
    
    print("\nTop citing journals:")
    for journal in citation_data.get('top_citing_journals', [])[:5]:
        print(f"  {journal.get('journal')}: {journal.get('count')} citations")
    
    # Close the connection
    analyzer.close()