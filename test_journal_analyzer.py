#!/usr/bin/env python3
"""
Test script for the Journal Analyzer and Scopus API integration.

This script tests the functionality of the JournalAnalyzer class,
which integrates with the Scopus API to retrieve and analyze
journal information for the paper revision process.
"""

import os
import sys
import logging
import json
from src.utils.journal_analyzer import get_journal_analyzer
from src.utils.workflow_db import WorkflowDB

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_journal_analyzer")

# Scopus API key
SCOPUS_API_KEY = os.environ.get("SCOPUS_API_KEY", "794f87fe4933b144dd95702b217fcb50")

def test_journal_analyzer():
    """Test the Journal Analyzer functionality."""
    logger.info("Testing Journal Analyzer...")
    
    # Create analyzer
    analyzer = get_journal_analyzer(scopus_api_key=SCOPUS_API_KEY)
    
    # Test analyzing a journal by ISSN (Computers in Human Behavior)
    logger.info("Testing analyze_journal_by_issn...")
    journal = analyzer.analyze_journal_by_issn("0747-5632")
    
    if journal and "journal" in journal:
        journal_info = journal["journal"]
        logger.info(f"Successfully analyzed journal: {journal_info.get('title', '')}")
        logger.info(f"Publisher: {journal_info.get('publisher', '')}")
        logger.info(f"Subject areas: {journal_info.get('subject_areas', '')}")
        
        if "metrics" in journal:
            metrics = journal["metrics"]
            logger.info(f"SJR: {metrics.get('sjr', '')}")
            logger.info(f"SNIP: {metrics.get('snip', '')}")
            logger.info(f"Impact Factor: {metrics.get('impact_factor', '')}")
        
        if "top_papers" in journal:
            top_papers = journal["top_papers"]
            logger.info(f"Top papers: {len(top_papers)}")
            for i, paper in enumerate(top_papers[:3], 1):
                logger.info(f"{i}. {paper.get('title', '')} - Citations: {paper.get('citations', 0)}")
    else:
        logger.warning("Failed to analyze journal by ISSN")
    
    # Test analyzing a journal by title
    logger.info("\nTesting analyze_journal_by_title...")
    journal_by_title = analyzer.analyze_journal_by_title("Computers in Human Behavior")
    
    if journal_by_title and "journal" in journal_by_title:
        journal_info = journal_by_title["journal"]
        logger.info(f"Successfully analyzed journal: {journal_info.get('title', '')}")
        logger.info(f"ISSN: {journal_info.get('issn', '')}")
    else:
        logger.warning("Failed to analyze journal by title")
    
    # Test finding similar papers
    if journal and "journal" in journal:
        journal_id = journal_info.get("journal_id", 0)
        
        logger.info("\nTesting find_similar_papers...")
        test_title = "Machine learning for human behavior analysis"
        test_abstract = """
        This paper explores the application of machine learning techniques for analyzing and
        predicting human behavior in digital environments. We examine various methods
        including deep learning and statistical approaches to understand patterns in human-computer
        interaction and social media behavior.
        """
        
        similar_papers = analyzer.find_similar_papers("test_run_id", journal_id, test_title, test_abstract)
        
        if similar_papers:
            logger.info(f"Found {len(similar_papers)} similar papers")
            for i, paper in enumerate(similar_papers[:3], 1):
                logger.info(f"{i}. {paper.get('title', '')} - Similarity: {paper.get('similarity_score', 0):.2f}")
                logger.info(f"   Reason: {paper.get('similarity_reason', '')}")
        else:
            logger.warning("No similar papers found")
        
        # Test getting journal guidance
        logger.info("\nTesting get_journal_guidance...")
        guidance = analyzer.get_journal_guidance(journal_id, test_title, test_abstract)
        
        if guidance:
            logger.info("Journal guidance:")
            logger.info(f"Align with scope: {guidance.get('suggestions', {}).get('align_with_scope', False)}")
            logger.info("Key topics:")
            for topic in guidance.get('suggestions', {}).get('key_topics', []):
                logger.info(f"- {topic}")
        else:
            logger.warning("Failed to get journal guidance")
    
    # Close the analyzer
    analyzer.close()
    logger.info("Journal Analyzer tests completed")
    
    # Print database summary
    logger.info("\nDatabase summary:")
    db = WorkflowDB()
    
    db.cursor.execute("SELECT COUNT(*) FROM journals")
    journal_count = db.cursor.fetchone()[0]
    logger.info(f"Journals: {journal_count}")
    
    db.cursor.execute("SELECT COUNT(*) FROM journal_metrics")
    metrics_count = db.cursor.fetchone()[0]
    logger.info(f"Journal metrics: {metrics_count}")
    
    db.cursor.execute("SELECT COUNT(*) FROM journal_top_papers")
    papers_count = db.cursor.fetchone()[0]
    logger.info(f"Top papers: {papers_count}")
    
    db.cursor.execute("SELECT COUNT(*) FROM journal_preferences")
    prefs_count = db.cursor.fetchone()[0]
    logger.info(f"Journal preferences: {prefs_count}")
    
    db.cursor.execute("SELECT COUNT(*) FROM similar_journal_papers")
    similar_count = db.cursor.fetchone()[0]
    logger.info(f"Similar papers: {similar_count}")
    
    db.close()

if __name__ == "__main__":
    try:
        test_journal_analyzer()
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)