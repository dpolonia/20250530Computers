#!/usr/bin/env python3
"""
Paper Revision Tool

This script helps researchers revise academic papers based on reviewer comments
using LLM-powered analysis and suggestions.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

from src.config import (
    AppConfig, load_env_files, load_config, initialize_logging,
    create_argument_parser, parse_args, process_args, validate_args,
    run_with_args
)
from src.config.manager import get_manager
from src.core.paper_revision_tool import PaperRevisionTool
from src.errors import (
    PaperRevisionError, FileError, LLMError, ValidationError,
    ErrorHandler, ConsoleReporter, FileReporter, LoggingReporter,
    CompositeReporter, handle_errors, ErrorHandlerContext,
    create_default_reporter, set_reporter
)


@handle_errors(recover=False)
def run_revision_tool(config: AppConfig) -> int:
    """
    Run the paper revision tool with the given configuration.
    
    Args:
        config: Configuration for the paper revision tool
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Configure logging
    logger = logging.getLogger("PaperRevision")
    
    # Initialize the paper revision tool with the configuration
    tool = PaperRevisionTool(
        original_paper_path=config.files.original_paper_path,  # Still needed as constructor param
        reviewer_comment_files=config.files.reviewer_comment_files,  # Still needed as constructor param
        verbose=config.logging.level.lower() == "debug",
        config=config
    )
    
    # Run in full mode or interactive mode
    if not config.interactive:
        # Full mode
        logger.info("Running paper revision process in full mode")
        
        try:
            created_docs = tool.run_full_process()
            
            logger.info("\nPaper revision process completed successfully")
            logger.info("Created documents:")
            for doc_type, path in created_docs.items():
                logger.info(f"- {doc_type}: {path}")
            
            # Log process statistics
            logger.info("\nProcess statistics:")
            logger.info(f"Total tokens used: {tool.context.process_statistics.get('total_tokens', 0)}")
            logger.info(f"Total cost: ${tool.context.process_statistics.get('total_cost', 0.0):.4f}")
            
            return 0
            
        except PaperRevisionError as e:
            logger.error(f"Error running paper revision process: {e}")
            return 1
    
    else:
        # Interactive mode
        print("\nPaper Revision Tool")
        print("===================")
        print("\nAvailable commands:")
        print("1. Analyze paper")
        print("2. Analyze reviewer comments")
        print("3. Analyze editor requirements")
        print("4. Identify issues")
        print("5. Generate solutions")
        print("6. Generate specific changes")
        print("7. Validate and update references")
        print("8. Create changes document")
        print("9. Create revised paper")
        print("10. Create assessment document")
        print("11. Create editor letter")
        print("12. Run full process")
        print("0. Exit")
        
        while True:
            try:
                choice = input("\nEnter command number: ").strip()
                
                if choice == "0":
                    print("Exiting...")
                    break
                    
                # Use ErrorHandlerContext for each interactive command
                with ErrorHandlerContext(recover=False, context={"command": choice}) as context:
                    if choice == "1":
                        paper = tool.analyze_paper()
                        print(f"Paper analyzed: {paper.title}")
                        print(f"Sections: {len(paper.sections)}")
                        print(f"References: {len(paper.references)}")
                        
                    elif choice == "2":
                        reviewer_comments = tool.analyze_reviewer_comments()
                        print(f"Analyzed {len(reviewer_comments)} reviewer comments")
                        
                    elif choice == "3":
                        editor_requirements = tool.analyze_editor_requirements()
                        if editor_requirements:
                            print(f"Editor decision: {editor_requirements.get('decision', 'N/A')}")
                            print(f"Key requirements: {len(editor_requirements.get('key_requirements', []))}")
                        else:
                            print("No editor requirements found")
                        
                    elif choice == "4":
                        issues = tool.identify_issues()
                        print(f"Identified {len(issues)} issues")
                        
                    elif choice == "5":
                        solutions = tool.generate_solutions()
                        print(f"Generated {len(solutions)} solutions")
                        
                    elif choice == "6":
                        changes = tool.generate_specific_changes()
                        print(f"Generated {len(changes)} specific changes")
                        
                    elif choice == "7":
                        new_references = tool.validate_and_update_references()
                        print(f"Added {len(new_references)} new references")
                        
                    elif choice == "8":
                        output_path = input("Enter output path (leave blank for default): ").strip()
                        output_path = output_path if output_path else None
                        changes_doc = tool.create_changes_document(output_path)
                        print(f"Changes document created: {changes_doc}")
                        
                    elif choice == "9":
                        output_path = input("Enter output path (leave blank for default): ").strip()
                        output_path = output_path if output_path else None
                        revised_paper = tool.create_revised_paper(output_path)
                        print(f"Revised paper created: {revised_paper}")
                        
                    elif choice == "10":
                        output_path = input("Enter output path (leave blank for default): ").strip()
                        output_path = output_path if output_path else None
                        assessment_doc = tool.create_assessment(output_path)
                        print(f"Assessment document created: {assessment_doc}")
                        
                    elif choice == "11":
                        output_path = input("Enter output path (leave blank for default): ").strip()
                        output_path = output_path if output_path else None
                        editor_letter = tool.create_editor_letter(output_path)
                        print(f"Editor letter created: {editor_letter}")
                        
                    elif choice == "12":
                        created_docs = tool.run_full_process()
                        print("Paper revision process completed successfully")
                        print("Created documents:")
                        for doc_type, path in created_docs.items():
                            print(f"- {doc_type}: {path}")
                    
                    elif choice != "0":
                        print("Invalid command number")
                
            except PaperRevisionError as e:
                # Errors are already handled by the ErrorHandlerContext
                pass
        
        # Log final statistics
        logger.info("\nFinal statistics:")
        logger.info(f"Total tokens used: {tool.context.process_statistics.get('total_tokens', 0)}")
        logger.info(f"Total cost: ${tool.context.process_statistics.get('total_cost', 0.0):.4f}")
        
        return 0


def setup_error_handling(config: AppConfig) -> None:
    """
    Set up error handling based on configuration.
    
    Args:
        config: Application configuration
    """
    # Create reporters based on configuration
    reporters = []
    
    # Always add console reporter
    reporters.append(ConsoleReporter(
        show_traceback=config.logging.level.lower() in ["debug", "trace"],
        color=not config.logging.disable_colors
    ))
    
    # Add file reporter if error logging is enabled
    if config.logging.error_log_file:
        reporters.append(FileReporter(
            file_path=config.logging.error_log_file,
            show_traceback=True,
            include_timestamp=True
        ))
    
    # Add logging reporter
    logger = logging.getLogger("PaperRevision.Errors")
    reporters.append(LoggingReporter(logger=logger))
    
    # Create and set composite reporter
    set_reporter(CompositeReporter(reporters))


def main():
    """Main function to run the paper revision tool."""
    # Load environment variables
    load_env_files()
    
    # Initialize configuration manager
    manager = get_manager()
    
    try:
        # Load configuration from command-line arguments
        config = manager.load_from_args()
        
        # Initialize logging
        initialize_logging(config.logging)
        
        # Set up error handling
        setup_error_handling(config)
        
        # Print configuration summary
        manager.print_summary()
        
        # Run the revision tool with the configuration
        return run_revision_tool(config)
        
    except ValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except FileError as e:
        print(f"File error: {e}")
        return 1
    except Exception as e:
        # Convert standard exceptions to PaperRevisionError
        from src.errors.exceptions import parse_exception
        error = parse_exception(e)
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())