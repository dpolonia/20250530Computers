#!/usr/bin/env python3
"""
Test script for Scopus API integration.

This script tests the Scopus API client and its integration with the ReferenceValidator.
"""

import os
import sys
import logging
import json
from src.utils.scopus_client import get_scopus_client
from src.utils.reference_validator import ReferenceValidator

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_scopus")

def test_scopus_client():
    """Test the Scopus API client functionality."""
    logger.info("Testing Scopus API client...")
    
    # Create a client with the API key from the environment or provided value
    api_key = os.environ.get("SCOPUS_API_KEY", "794f87fe4933b144dd95702b217fcb50")
    client = get_scopus_client(api_key=api_key)
    
    # Test paper search by title
    logger.info("Testing search by title...")
    papers = client.search_by_title("Machine Learning")
    logger.info(f"Found {len(papers)} papers matching the title 'Machine Learning'")
    
    if papers:
        paper = papers[0]
        logger.info(f"First result: {paper.get('dc:title', '')}")
        logger.info(f"Authors: {paper.get('dc:creator', '')}")
        logger.info(f"Journal: {paper.get('prism:publicationName', '')}")
        logger.info(f"Citations: {paper.get('citedby-count', '0')}")
        
        # Test DOI search
        if 'prism:doi' in paper:
            doi = paper['prism:doi']
            logger.info(f"Testing search by DOI: {doi}")
            paper_by_doi = client.search_by_doi(doi)
            if paper_by_doi:
                logger.info(f"Found paper by DOI: {paper_by_doi.get('dc:title', '')}")
            else:
                logger.error(f"Failed to find paper by DOI: {doi}")
    else:
        logger.warning("No papers found by title search")
    
    # Test similar paper recommendations
    logger.info("Testing similar paper recommendations...")
    if papers:
        similar_papers = client.recommend_similar_papers(
            papers[0].get('dc:title', ''),
            papers[0].get('dc:description', ''),
            count=3
        )
        logger.info(f"Found {len(similar_papers)} similar papers")
        for i, paper in enumerate(similar_papers, 1):
            logger.info(f"{i}. {paper.get('dc:title', '')}")
    
    # Test citation report
    logger.info("Testing citation report generation...")
    if papers and 'prism:doi' in papers[0]:
        doi = papers[0]['prism:doi']
        report = client.generate_citation_report(doi)
        logger.info(f"Citation report for DOI {doi}:")
        logger.info(f"Total citations: {report['citations']['total']}")
        logger.info(f"Recent citations: {report['citations']['recent']}")
    
    logger.info("Scopus client tests completed")

def test_reference_validator():
    """Test the reference validator with Scopus integration."""
    logger.info("Testing reference validator with Scopus integration...")
    
    # Create a temporary BibTeX file
    bib_path = "./test_refs.bib"
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write("""@article{Smith2020,
  author = {Smith, John},
  title = {Machine Learning Applications},
  journal = {Journal of AI},
  year = {2020},
  volume = {10},
  number = {2},
  pages = {123--145}
}

@article{Jones2021,
  author = {Jones, Sarah},
  title = {Deep Learning in Computer Vision},
  journal = {Vision Research},
  year = {2021},
  doi = {10.1000/invalid-doi}
}
""")
    
    # Initialize the reference validator
    validator = ReferenceValidator(bib_path, use_scopus=True)
    
    # Test reference validation
    logger.info("Testing reference validation...")
    valid_refs, invalid_refs = validator.validate_references(fix_invalid=True)
    logger.info(f"Valid references: {len(valid_refs)}")
    logger.info(f"Invalid references: {len(invalid_refs)}")
    
    # Test finding similar papers
    logger.info("Testing finding similar papers...")
    if valid_refs:
        ref_id = list(valid_refs)[0]
        similar_papers = validator.find_similar_papers(ref_id, count=3)
        logger.info(f"Found {len(similar_papers)} similar papers")
        for i, paper in enumerate(similar_papers, 1):
            logger.info(f"{i}. {paper.get('title', '')}")
    
    # Test adding reference from DOI
    logger.info("Testing adding reference from DOI...")
    test_doi = "10.1016/j.artint.2022.103756"  # Example DOI
    ref_id = validator.add_reference_from_doi(test_doi)
    if ref_id:
        logger.info(f"Added reference with ID: {ref_id}")
        logger.info(f"Title: {validator.references[ref_id].get('title', '')}")
    else:
        logger.warning(f"Failed to add reference from DOI: {test_doi}")
    
    # Test searching and adding references
    logger.info("Testing searching and adding references...")
    added_refs = validator.search_and_add_references("Natural Language Processing", count=2)
    logger.info(f"Added {len(added_refs)} references from search")
    for ref_id in added_refs:
        logger.info(f"- {validator.references[ref_id].get('title', '')}")
    
    # Save the updated references
    validator.save_references("./updated_refs.bib")
    logger.info("Saved updated references to ./updated_refs.bib")
    
    # Clean up
    if os.path.exists(bib_path):
        os.remove(bib_path)
    if os.path.exists("./updated_refs.bib"):
        os.remove("./updated_refs.bib")
    
    logger.info("Reference validator tests completed")

if __name__ == "__main__":
    try:
        test_scopus_client()
        test_reference_validator()
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)