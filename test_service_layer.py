"""
Test script for the service layer.

This script demonstrates the usage of the service layer to revise a paper
based on reviewer comments.
"""

import os
import logging
import argparse
from typing import List, Optional

from src.core.paper_revision_tool import PaperRevisionTool


def main():
    """Main function to run the paper revision tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Paper Revision Tool')
    parser.add_argument('--paper', required=True, help='Path to the original paper PDF')
    parser.add_argument('--reviewer_comments', nargs='+', required=True, help='Paths to reviewer comment files')
    parser.add_argument('--editor_letter', help='Path to editor letter file')
    parser.add_argument('--output_dir', help='Directory for output files')
    parser.add_argument('--provider', default='anthropic', help='LLM provider (default: anthropic)')
    parser.add_argument('--model', default='claude-3-opus-20240229', help='LLM model name (default: claude-3-opus-20240229)')
    parser.add_argument('--optimize_costs', action='store_true', help='Optimize for lower costs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("TestServiceLayer")
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize the paper revision tool
    tool = PaperRevisionTool(
        original_paper_path=args.paper,
        reviewer_comment_files=args.reviewer_comments,
        editor_letter_path=args.editor_letter,
        output_dir=args.output_dir,
        provider=args.provider,
        model_name=args.model,
        optimize_costs=args.optimize_costs,
        verbose=args.verbose
    )
    
    # Run the full process
    logger.info("Starting paper revision process")
    
    try:
        # Run individual steps to demonstrate the service layer
        logger.info("Step 1: Analyzing paper")
        paper = tool.analyze_paper()
        logger.info(f"Paper title: {paper.title}")
        logger.info(f"Paper sections: {len(paper.sections)}")
        
        logger.info("\nStep 2: Analyzing reviewer comments")
        reviewer_comments = tool.analyze_reviewer_comments()
        logger.info(f"Number of reviewers: {len(reviewer_comments)}")
        
        logger.info("\nStep 3: Analyzing editor requirements")
        editor_requirements = tool.analyze_editor_requirements()
        if editor_requirements:
            logger.info(f"Editor decision: {editor_requirements.get('decision', 'N/A')}")
        
        logger.info("\nStep 4: Identifying issues")
        issues = tool.identify_issues()
        logger.info(f"Number of issues identified: {len(issues)}")
        
        logger.info("\nStep 5: Generating solutions")
        solutions = tool.generate_solutions()
        logger.info(f"Number of solutions generated: {len(solutions)}")
        
        logger.info("\nStep 6: Generating specific text changes")
        changes = tool.generate_specific_changes()
        logger.info(f"Number of specific changes: {len(changes)}")
        
        logger.info("\nStep 7: Validating and updating references")
        new_references = tool.validate_and_update_references()
        logger.info(f"Number of new references: {len(new_references)}")
        
        logger.info("\nStep 8: Creating output documents")
        changes_doc = tool.create_changes_document()
        logger.info(f"Changes document created: {changes_doc}")
        
        revised_paper = tool.create_revised_paper()
        logger.info(f"Revised paper created: {revised_paper}")
        
        assessment_doc = tool.create_assessment()
        logger.info(f"Assessment document created: {assessment_doc}")
        
        editor_letter = tool.create_editor_letter()
        logger.info(f"Editor letter created: {editor_letter}")
        
        # Log process statistics
        logger.info("\nProcess statistics:")
        logger.info(f"Total tokens used: {tool.context.process_statistics.get('total_tokens', 0)}")
        logger.info(f"Total cost: ${tool.context.process_statistics.get('total_cost', 0.0):.4f}")
        
    except Exception as e:
        logger.error(f"Error running paper revision process: {e}", exc_info=True)
        return 1
    
    logger.info("Paper revision process completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())