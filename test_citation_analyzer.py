#!/usr/bin/env python3
"""
Test script for the Citation Analyzer with multi-database integration.

This script demonstrates the functionality of the CitationAnalyzer class,
which integrates data from both Scopus and Web of Science APIs to provide
comprehensive citation analysis for academic papers.
"""

import os
import sys
import logging
import json
import datetime
from src.utils.citation_analyzer import get_citation_analyzer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_citation_analyzer")

# API credentials
SCOPUS_API_KEY = os.environ.get("SCOPUS_API_KEY", "794f87fe4933b144dd95702b217fcb50")
# WoS credentials are commented out as WoS integration is currently inactive
# WOS_CLIENT_ID = os.environ.get("WOS_CLIENT_ID", "lazyresearcher")
# WOS_CLIENT_SECRET = os.environ.get("WOS_CLIENT_SECRET", "")

def test_citation_analyzer():
    """Test the Citation Analyzer functionality."""
    logger.info("Testing Citation Analyzer...")
    
    # Create analyzer with Scopus credentials only (WoS integration inactive)
    analyzer = get_citation_analyzer(
        scopus_api_key=SCOPUS_API_KEY
        # WoS credentials omitted as integration is currently inactive
    )
    
    # Test paper citation analysis
    logger.info("Testing analyze_paper_citations...")
    doi = "10.1016/j.cose.2022.102644"  # Example paper DOI
    
    citation_data = analyzer.analyze_paper_citations(doi)
    
    if citation_data:
        logger.info(f"Citation analysis for DOI: {citation_data.get('doi')}")
        logger.info(f"Title: {citation_data.get('title')}")
        logger.info(f"Sources: {', '.join(citation_data.get('sources', []))}")
        logger.info(f"Total citations: {citation_data.get('total_citations', 0)}")
        
        for source, count in citation_data.get('citation_counts', {}).items():
            logger.info(f"  {source.upper()}: {count}")
        
        logger.info("\nTop citing journals:")
        for journal in citation_data.get('top_citing_journals', [])[:3]:
            logger.info(f"  {journal.get('journal')}: {journal.get('count')} citations")
        
        logger.info(f"Citation velocity: {citation_data.get('citation_velocity', 0):.2f} citations/year")
    else:
        logger.warning("Failed to get citation data")
    
    # Test cross-database paper search
    logger.info("\nTesting find_cross_database_papers...")
    test_title = "Machine learning for cybersecurity"
    test_abstract = """
    This paper explores the application of machine learning techniques for cybersecurity.
    We examine various methods including deep learning and statistical approaches to
    detect and prevent cyber attacks, malware, and other security threats.
    """
    
    similar_papers = analyzer.find_cross_database_papers(test_title, test_abstract, count=5)
    
    if similar_papers:
        logger.info(f"Found {len(similar_papers)} papers across databases")
        for i, paper in enumerate(similar_papers, 1):
            logger.info(f"{i}. {paper.get('title')} ({paper.get('year')})")
            logger.info(f"   Source: {paper.get('source').upper()}, Citations: {paper.get('citations')}")
            logger.info(f"   DOI: {paper.get('doi')}")
    else:
        logger.warning("No similar papers found")
    
    # Test citation overlap analysis
    logger.info("\nTesting get_citation_overlap...")
    dois = [
        "10.1016/j.cose.2022.102644",
        "10.1016/j.diin.2022.301504"
    ]
    
    overlap_data = analyzer.get_citation_overlap(dois)
    
    if overlap_data:
        logger.info(f"Citation overlap analysis for {len(overlap_data.get('papers', []))} papers")
        logger.info(f"Total unique citations: {overlap_data.get('total_unique_citations', 0)}")
        
        common_citations = overlap_data.get('common_citations', {})
        logger.info(f"Citations common to all papers: {common_citations.get('common_to_all', 0)}")
        logger.info(f"Citations common to at least two papers: {common_citations.get('common_to_some', 0)}")
        
        logger.info("\nOverlap matrix:")
        overlap_matrix = overlap_data.get('overlap_matrix', {})
        for doi1 in dois:
            for doi2 in dois:
                if doi1 in overlap_matrix and doi2 in overlap_matrix[doi1]:
                    logger.info(f"  {doi1[-8:]} <-> {doi2[-8:]}: {overlap_matrix[doi1][doi2]:.2f}")
    else:
        logger.warning("Failed to get citation overlap data")
    
    # Get usage statistics
    logger.info("\nAPI Usage Statistics:")
    stats = analyzer.get_usage_statistics()
    
    if "scopus_stats" in stats:
        scopus_stats = stats["scopus_stats"]
        logger.info(f"Scopus API Key: ...{scopus_stats.get('api_key', '')}")
        logger.info(f"Cache entries: {scopus_stats.get('cache_entries', 0)}")
        logger.info(f"Cache size: {scopus_stats.get('cache_size_mb', 0):.2f} MB")
        logger.info(f"API requests: {scopus_stats.get('api_requests', 0)}")
        logger.info(f"Cache hits: {scopus_stats.get('cache_hits', 0)}")
        logger.info(f"Cache hit rate: {scopus_stats.get('cache_hit_rate', 0):.2f}%")
    
    if "wos_stats" in stats:
        wos_stats = stats["wos_stats"]
        logger.info(f"\nWeb of Science Client ID: {wos_stats.get('client_id', '')}")
        logger.info(f"Token valid: {wos_stats.get('token_valid', False)}")
        logger.info(f"Token expires in: {wos_stats.get('token_expires_in', 0):.0f} seconds")
        logger.info(f"Cache entries: {wos_stats.get('cache_entries', 0)}")
        logger.info(f"API requests: {wos_stats.get('api_requests', 0)}")
    
    # Export results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"citation_analysis_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "citation_analysis": citation_data,
            "similar_papers": similar_papers,
            "citation_overlap": overlap_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "api_stats": stats
        }, f, indent=2)
    
    logger.info(f"\nResults exported to {filename}")
    
    # Close the analyzer
    analyzer.close()
    logger.info("Citation Analyzer tests completed")

if __name__ == "__main__":
    try:
        test_citation_analyzer()
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)