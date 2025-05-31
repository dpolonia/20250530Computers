"""
Main entry point for the paper revision tool.
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime

from colorama import init as colorama_init, Fore, Style

from src.core.context import RevisionContext
from src.core.constants import OPERATION_MODES
from src.core.interactive import interactive_wait
from src.budget.budget_manager import BudgetManager
from src.utils.llm_client import get_llm_client, get_model_choices


class PaperRevisionTool:
    """
    Main entry point for the paper revision tool.
    
    This class orchestrates the entire paper revision process, delegating
    specialized tasks to the appropriate modules.
    """
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        original_paper_path: str,
        reviewer1_path: str,
        reviewer2_path: str,
        reviewer3_path: str,
        editor_letter_path: str,
        prisma_requirements_path: str,
        output_dir: str,
        operation_mode: str = "finetuning",
        optimize_costs: bool = False,
        budget: float = 10.0,
        use_cache: bool = True,
        competitor_evaluation: bool = True,
        competing_evaluator: Optional[str] = None,
        interactive: bool = False,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the paper revision tool.
        
        Args:
            provider: LLM provider (anthropic, openai, google)
            model_name: Model to use
            original_paper_path: Path to the original paper PDF
            reviewer1_path: Path to the first reviewer comments PDF
            reviewer2_path: Path to the second reviewer comments PDF
            reviewer3_path: Path to the third reviewer comments PDF
            editor_letter_path: Path to the editor letter PDF
            prisma_requirements_path: Path to the PRISMA requirements PDF
            output_dir: Directory where output files will be saved
            operation_mode: Operation mode (training, finetuning, final)
            optimize_costs: Whether to optimize costs by using smaller models when possible
            budget: Maximum budget in dollars
            use_cache: Whether to use LLM response caching
            competitor_evaluation: Whether to use a competing model for evaluation
            competing_evaluator: Specific competing model to use for evaluation (provider/model)
            interactive: Whether to run in interactive mode with wait points
            run_id: Optional run ID for tracking in workflow DB
        """
        # Create the context
        self.context = RevisionContext(
            provider=provider,
            model_name=model_name,
            original_paper_path=original_paper_path,
            reviewer1_path=reviewer1_path,
            reviewer2_path=reviewer2_path,
            reviewer3_path=reviewer3_path,
            editor_letter_path=editor_letter_path,
            prisma_requirements_path=prisma_requirements_path,
            output_dir=output_dir,
            operation_mode=operation_mode,
            optimize_costs=optimize_costs,
            budget=budget,
            use_cache=use_cache,
            competitor_evaluation=competitor_evaluation,
            competing_evaluator=competing_evaluator,
            interactive=interactive,
            run_id=run_id,
        )
        
        # Initialize colorama
        colorama_init()
        
        # Setup logger first (used by budget and other modules)
        self.logger = self.context.setup_logger()
        
        # Setup budget manager
        self.budget_manager = BudgetManager(
            budget=budget,
            logger=self.logger,
            statistics=self.context.process_statistics
        )
        
        # Setup LLM client
        self.llm_client = self.context.setup_llm_client()
        
        # Log initialization
        self.logger.info(f"Initialized paper revision tool with {provider}/{model_name}")
        self.logger.info(f"Operation mode: {operation_mode}")
        self.logger.info(f"Budget: ${budget:.2f}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Log interactive mode status
        if interactive:
            self.logger.info("Interactive mode enabled with wait points")
        
        # Initialize workflow database connection if run_id is provided
        self.workflow_db = None
        if run_id:
            try:
                from src.utils.workflow_db import WorkflowDB
                self.workflow_db = WorkflowDB()
                self.logger.info(f"Connected to workflow database with run ID: {run_id}")
            except ImportError:
                self.logger.warning("WorkflowDB module not found. Run tracking disabled.")
            except Exception as e:
                self.logger.error(f"Error connecting to workflow database: {e}")
    
    def _interactive_wait(self, message: str, path: Optional[str] = None) -> None:
        """
        Wait for user input in interactive mode.
        
        Args:
            message: Message to display
            path: Optional path to show to the user
        """
        interactive_wait(message, path, self.context.interactive)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the paper revision process.
        
        Returns:
            Dictionary with process statistics
        """
        self.logger.info("Starting paper revision process")
        
        # Step 1: Analyze the original paper
        # (Will be implemented in analysis module)
        self._step_1_analyze_paper()
        
        # Step 2: Analyze reviewer comments
        # (Will be implemented in analysis module)
        self._step_2_analyze_reviewer_comments()
        
        # Step 3: Process editor letter and requirements
        # (Will be implemented in analysis module)
        self._step_3_process_editor_requirements()
        
        # Step 4: Identify issues and reviewer concerns
        # (Will be implemented in analysis module)
        self._step_4_identify_issues()
        
        # Step 5: Generate solutions for identified issues
        # (Will be implemented in analysis module)
        self._step_5_generate_solutions()
        
        # Step 6: Generate specific changes
        # (Will be implemented in document module)
        self._step_6_generate_specific_changes()
        
        # Step 7: Create changes document
        # (Will be implemented in document module)
        self._step_7_create_changes_document()
        
        # Step 8: Validate and update references
        # (Will be implemented in references module)
        self._step_8_validate_and_update_references()
        
        # Step 9: Create revised paper
        # (Will be implemented in document module)
        self._step_9_create_revised_paper()
        
        # Step 10: Create assessment and editor letter
        # (Will be implemented in document module)
        self._step_10_create_assessment_and_letter()
        
        # Finalize and clean up
        return self._finalize()
    
    def _step_1_analyze_paper(self) -> None:
        """Step 1: Analyze the original paper."""
        self.logger.info("Step 1: Analyzing original paper")
        
        # Use factory to create the paper analyzer
        from src.core.factory import create_paper_analyzer
        paper_analyzer = create_paper_analyzer(self.context)
        self.context.paper_analysis = paper_analyzer.analyze()
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 1 completed: Analyzed the original paper.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_2_analyze_reviewer_comments(self) -> None:
        """Step 2: Analyze reviewer comments."""
        self.logger.info("Step 2: Analyzing reviewer comments")
        
        # Use factory to create the reviewer analyzer
        from src.core.factory import create_reviewer_analyzer
        reviewer_analyzer = create_reviewer_analyzer(self.context)
        self.context.reviewer_comments = reviewer_analyzer.analyze_reviewer_comments()
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 2 completed: Analyzed reviewer comments.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_3_process_editor_requirements(self) -> None:
        """Step 3: Process editor letter and requirements."""
        self.logger.info("Step 3: Processing editor letter and requirements")
        
        # Use factory to create the reviewer analyzer
        from src.core.factory import create_reviewer_analyzer
        reviewer_analyzer = create_reviewer_analyzer(self.context)
        self.context.editor_requirements = reviewer_analyzer.analyze_editor_requirements()
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 3 completed: Processed editor letter and requirements.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_4_identify_issues(self) -> None:
        """Step 4: Identify issues and reviewer concerns."""
        self.logger.info("Step 4: Identifying issues and reviewer concerns")
        
        # Use SolutionGenerator to identify issues
        from src.analysis import SolutionGenerator
        solution_generator = SolutionGenerator(self.context)
        self.context.issues = solution_generator.identify_issues(
            self.context.paper_analysis,
            self.context.reviewer_comments,
            self.context.editor_requirements
        )
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                f"Step 4 completed: Identified {len(self.context.issues)} issues and reviewer concerns.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_5_generate_solutions(self) -> None:
        """Step 5: Generate solutions for identified issues."""
        self.logger.info("Step 5: Generating solutions for identified issues")
        
        # Use SolutionGenerator to generate solutions
        from src.analysis import SolutionGenerator
        solution_generator = SolutionGenerator(self.context)
        self.context.solutions = solution_generator.generate_solutions(
            self.context.paper_analysis,
            self.context.issues
        )
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                f"Step 5 completed: Generated {len(self.context.solutions)} solutions for identified issues.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_6_generate_specific_changes(self) -> None:
        """Step 6: Generate specific changes."""
        self.logger.info("Step 6: Generating specific changes")
        
        # Use SolutionGenerator to generate specific changes
        from src.analysis import SolutionGenerator
        solution_generator = SolutionGenerator(self.context)
        self.context.changes = solution_generator.generate_specific_changes(
            self.context.paper_analysis,
            self.context.solutions
        )
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                f"Step 6 completed: Generated {len(self.context.changes)} specific changes.",
                self.context.log_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_7_create_changes_document(self) -> None:
        """Step 7: Create changes document."""
        self.logger.info("Step 7: Creating changes document")
        
        # Use ChangesDocumentGenerator to create changes document
        from src.document import ChangesDocumentGenerator
        changes_generator = ChangesDocumentGenerator(self.context)
        changes_path = changes_generator.create_changes_document(self.context.changes)
        
        # Store the path to the changes document
        self.context.changes_document_path = changes_path
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 7 completed: Created changes document.",
                changes_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_8_validate_and_update_references(self) -> None:
        """Step 8: Validate and update references."""
        self.logger.info("Step 8: Validating and updating references")
        
        # Use ReferenceManager to validate and update references
        from src.references import ReferenceManager
        reference_manager = ReferenceManager(self.context)
        new_references = reference_manager.validate_and_update_references(
            self.context.paper_analysis,
            self.context.reviewer_comments
        )
        
        # Store the new references
        self.context.new_references = new_references
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                f"Step 8 completed: Validated and updated references. Added {len(new_references)} new references.",
                self.context.get_output_path("references.bib")
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_9_create_revised_paper(self) -> None:
        """Step 9: Create revised paper."""
        self.logger.info("Step 9: Creating revised paper")
        
        # Use RevisedPaperGenerator to create revised paper
        from src.document import RevisedPaperGenerator
        paper_generator = RevisedPaperGenerator(self.context)
        revised_paper_path = paper_generator.create_revised_paper(self.context.changes)
        
        # Store the path to the revised paper
        self.context.revised_paper_path = revised_paper_path
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 9 completed: Created revised paper.",
                revised_paper_path
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _step_10_create_assessment_and_letter(self) -> None:
        """Step 10: Create assessment and editor letter."""
        self.logger.info("Step 10: Creating assessment and editor letter")
        
        # Use AssessmentGenerator to create assessment document
        from src.document import AssessmentGenerator
        assessment_generator = AssessmentGenerator(self.context)
        assessment_path = assessment_generator.create_assessment(
            self.context.changes,
            self.context.paper_analysis
        )
        
        # Store the path to the assessment document
        self.context.assessment_path = assessment_path
        
        # Use EditorLetterGenerator to create editor letter
        from src.document import EditorLetterGenerator
        letter_generator = EditorLetterGenerator(self.context)
        letter_path = letter_generator.create_editor_letter(
            self.context.reviewer_comments,
            self.context.changes
        )
        
        # Store the path to the editor letter
        self.context.editor_letter_path = letter_path
        
        # Interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Step 10 completed: Created assessment and editor letter.",
                f"Assessment: {assessment_path}\nEditor letter: {letter_path}"
            )
        
        # Mark step as completed
        self.context.complete_step()
    
    def _finalize(self) -> Dict[str, Any]:
        """
        Finalize the revision process and return statistics.
        
        Returns:
            Dictionary with process statistics
        """
        # Generate cost report
        cost_report = self.budget_manager.generate_report()
        cost_report_path = self.context.get_output_path("cost_report.txt")
        
        with open(cost_report_path, "w") as f:
            f.write(cost_report)
        
        self.logger.info(f"Cost report saved to {cost_report_path}")
        
        # Get statistics
        stats = self.budget_manager.get_statistics()
        
        # Print completion message
        print(f"\n{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}PAPER REVISION COMPLETED{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"Total cost: ${stats.get('total_cost', 0.0):.4f}")
        print(f"Total tokens: {stats.get('tokens_used', 0):,}")
        print(f"Output directory: {self.context.output_dir}")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        
        # If we have a workflow database connection, update the run
        if self.workflow_db and self.context.run_id:
            try:
                self.workflow_db.update_run_stats(self.context.run_id, stats)
                self.logger.info(f"Updated workflow database for run {self.context.run_id}")
            except Exception as e:
                self.logger.error(f"Error updating workflow database: {e}")
        
        # Close workflow database connection if open
        if self.workflow_db:
            try:
                self.workflow_db.close()
            except Exception as e:
                self.logger.error(f"Error closing workflow database: {e}")
        
        # Final interactive wait
        if self.context.interactive:
            self._interactive_wait(
                "Paper revision process completed. Check the output directory for results.",
                self.context.output_dir
            )
        
        return stats


def get_claude_model_choices():
    """Get choices for Claude models."""
    from src.models.anthropic_models import ANTHROPIC_MODELS
    return [f"{name}" for name in ANTHROPIC_MODELS.keys()]


def get_openai_model_choices():
    """Get choices for OpenAI models."""
    from src.models.openai_models import OPENAI_MODELS
    return [f"{name}" for name in OPENAI_MODELS.keys()]


def get_gemini_model_choices():
    """Get choices for Google Gemini models."""
    from src.models.google_models import GOOGLE_MODELS
    return [f"{name}" for name in GOOGLE_MODELS.keys()]


def choose_model(operation_mode=None, selected_provider=None):
    """Interactive model selection with default recommendations based on operation mode.
    
    Args:
        operation_mode: The operation mode to use for recommendations (training, finetuning, final)
        selected_provider: Optional pre-selected provider to skip provider selection
    
    Returns:
        Tuple of (provider, model)
    """
    colorama_init()
    
    # Get mode settings if provided
    mode_settings = OPERATION_MODES.get(operation_mode, {}) if operation_mode else {}
    provider_recommendations = mode_settings.get("provider_recommendations", {})
    
    # If provider is not pre-selected, ask user to select one
    if not selected_provider:
        print(f"{Fore.CYAN}Choose LLM Provider:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}1.{Style.RESET_ALL} Anthropic Claude" + 
              (f" {Fore.GREEN}[Recommended: {provider_recommendations.get('anthropic', '')}]{Style.RESET_ALL}" 
               if 'anthropic' in provider_recommendations else ""))
        print(f"{Fore.CYAN}2.{Style.RESET_ALL} OpenAI GPT" + 
              (f" {Fore.GREEN}[Recommended: {provider_recommendations.get('openai', '')}]{Style.RESET_ALL}" 
               if 'openai' in provider_recommendations else ""))
        print(f"{Fore.CYAN}3.{Style.RESET_ALL} Google Gemini" + 
              (f" {Fore.GREEN}[Recommended: {provider_recommendations.get('google', '')}]{Style.RESET_ALL}" 
               if 'google' in provider_recommendations else ""))
        
        provider_choice = input("Enter choice (1-3): ")
        
        if provider_choice == "1":
            provider = "anthropic"
        elif provider_choice == "2":
            provider = "openai"
        elif provider_choice == "3":
            provider = "google"
        else:
            print(f"{Fore.RED}Invalid choice. Defaulting to Anthropic Claude.{Style.RESET_ALL}")
            provider = "anthropic"
    else:
        provider = selected_provider
    
    # Get models based on provider
    if provider == "anthropic":
        models = get_claude_model_choices()
        recommended_model = provider_recommendations.get("anthropic")
    elif provider == "openai":
        models = get_openai_model_choices()
        recommended_model = provider_recommendations.get("openai")
    elif provider == "google":
        models = get_gemini_model_choices()
        recommended_model = provider_recommendations.get("google")
    else:
        print(f"{Fore.RED}Invalid provider. Defaulting to Anthropic Claude.{Style.RESET_ALL}")
        provider = "anthropic"
        models = get_claude_model_choices()
        recommended_model = provider_recommendations.get("anthropic")
    
    # Find the index of the recommended model if it exists
    recommended_index = -1
    if recommended_model:
        for i, model in enumerate(models):
            if recommended_model in model:
                recommended_index = i
                break
    
    print(f"\n{Fore.CYAN}Choose Model for {provider.capitalize()}:{Style.RESET_ALL}")
    
    # If we have a recommendation for this operation mode, highlight it prominently
    if operation_mode and recommended_index >= 0:
        print(f"{Fore.GREEN}Recommended model for {operation_mode.upper()} mode: {models[recommended_index]}{Style.RESET_ALL}")
        
    # List all models
    for i, model in enumerate(models, 1):
        model_text = f"{Fore.CYAN}{i}.{Style.RESET_ALL} {model}"
        # Highlight the recommended model
        if i - 1 == recommended_index:
            model_text += f" {Fore.GREEN}[RECOMMENDED]{Style.RESET_ALL}"
        print(model_text)
    
    # If there's a recommended model, use it as default
    if recommended_index >= 0 and operation_mode:
        model_choice = input(f"Enter choice (1-{len(models)}) or press Enter for recommended model: ")
        if not model_choice.strip():  # User pressed Enter
            return provider, models[recommended_index]
    else:
        model_choice = input(f"Enter choice (1-{len(models)}): ")
    
    try:
        if model_choice.strip():  # Only try to parse if not empty
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(models):
                chosen_model = models[model_index]
            else:
                if recommended_index >= 0:
                    print(f"{Fore.YELLOW}Invalid choice. Using recommended model.{Style.RESET_ALL}")
                    chosen_model = models[recommended_index]
                else:
                    print(f"{Fore.RED}Invalid choice. Defaulting to first model.{Style.RESET_ALL}")
                    chosen_model = models[0]
        elif recommended_index >= 0:  # Empty input with recommendation
            chosen_model = models[recommended_index]
        else:  # Empty input without recommendation
            print(f"{Fore.RED}No choice made. Defaulting to first model.{Style.RESET_ALL}")
            chosen_model = models[0]
    except ValueError:
        if recommended_index >= 0:
            print(f"{Fore.YELLOW}Invalid input. Using recommended model.{Style.RESET_ALL}")
            chosen_model = models[recommended_index]
        else:
            print(f"{Fore.RED}Invalid input. Defaulting to first model.{Style.RESET_ALL}")
            chosen_model = models[0]
    
    return provider, chosen_model


def choose_operation_mode():
    """Interactive operation mode selection."""
    colorama_init()
    
    print(f"{Fore.CYAN}Choose Operation Mode:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}1.{Style.RESET_ALL} {OPERATION_MODES['training']['description']}")
    print(f"   Recommended models: {Fore.GREEN}Anthropic: {OPERATION_MODES['training']['provider_recommendations']['anthropic']}, " +
          f"OpenAI: {OPERATION_MODES['training']['provider_recommendations']['openai']}, " +
          f"Google: {OPERATION_MODES['training']['provider_recommendations']['google']}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}2.{Style.RESET_ALL} {OPERATION_MODES['finetuning']['description']}")
    print(f"   Recommended models: {Fore.GREEN}Anthropic: {OPERATION_MODES['finetuning']['provider_recommendations']['anthropic']}, " +
          f"OpenAI: {OPERATION_MODES['finetuning']['provider_recommendations']['openai']}, " +
          f"Google: {OPERATION_MODES['finetuning']['provider_recommendations']['google']}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}3.{Style.RESET_ALL} {OPERATION_MODES['final']['description']}")
    print(f"   Recommended models: {Fore.GREEN}Anthropic: {OPERATION_MODES['final']['provider_recommendations']['anthropic']}, " +
          f"OpenAI: {OPERATION_MODES['final']['provider_recommendations']['openai']}, " +
          f"Google: {OPERATION_MODES['final']['provider_recommendations']['google']}{Style.RESET_ALL}")
    
    mode_choice = input("Enter choice (1-3): ")
    
    if mode_choice == "1":
        return "training"
    elif mode_choice == "2":
        return "finetuning"
    elif mode_choice == "3":
        return "final"
    else:
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Invalid choice. Defaulting to finetuning mode.")
        return "finetuning"


def main():
    """Main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(description="Paper Revision Tool")
    
    # Basic arguments
    parser.add_argument("--original", help="Path to original paper PDF", required=False)
    parser.add_argument("--reviewer1", help="Path to reviewer 1 comments PDF", required=False)
    parser.add_argument("--reviewer2", help="Path to reviewer 2 comments PDF", required=False)
    parser.add_argument("--reviewer3", help="Path to reviewer 3 comments PDF", required=False)
    parser.add_argument("--editor", help="Path to editor letter PDF", required=False)
    parser.add_argument("--prisma", help="Path to PRISMA requirements PDF", required=False)
    parser.add_argument("--output", help="Output directory", required=False, default="./tobe")
    
    # Model and provider arguments
    parser.add_argument("--provider", help="LLM provider (anthropic, openai, google)", 
                        choices=["anthropic", "openai", "google"], required=False)
    parser.add_argument("--model", help="Model name", required=False)
    
    # Operation mode and budget arguments
    parser.add_argument("--mode", help="Operation mode", 
                        choices=["training", "finetuning", "final"], default="finetuning")
    parser.add_argument("--budget", help="Maximum budget in dollars", 
                        type=float, default=None)
    parser.add_argument("--no-optimize", help="Disable cost optimization", 
                        action="store_true", default=False)
    parser.add_argument("--no-cache", help="Disable LLM response caching", 
                        action="store_true", default=False)
    parser.add_argument("--no-evaluation", help="Disable competitor evaluation", 
                        action="store_true", default=False)
    
    # Run tracking
    parser.add_argument("--run-id", help="Run ID for tracking in workflow DB", required=False)
    
    # Interactive mode
    parser.add_argument("--interactive", help="Enable interactive mode with wait points", 
                        action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Default paths (for demo purposes)
    default_paths = {
        "original": "./asis/00.pdf",
        "reviewer1": "./asis/01.pdf",
        "reviewer2": "./asis/02.pdf",
        "reviewer3": "./asis/03.pdf",
        "editor": "./asis/04.pdf",
        "prisma": "./asis/05.pdf",
    }
    
    # Use interactive mode to select provider and model if not provided
    if args.provider is None or args.model is None:
        # Choose operation mode if not using interactive mode for everything
        if args.mode == "finetuning" and not all([args.original, args.reviewer1, args.reviewer2, args.reviewer3, args.editor, args.prisma]):
            operation_mode = choose_operation_mode()
        else:
            operation_mode = args.mode
            
        # Choose provider and model
        provider, model = choose_model(operation_mode, args.provider)
    else:
        provider = args.provider
        model = args.model
        operation_mode = args.mode
    
    # Get operation mode settings
    mode_settings = OPERATION_MODES.get(operation_mode, {})
    optimize_costs = mode_settings.get("optimize_costs", True) if not args.no_optimize else False
    budget = args.budget if args.budget is not None else mode_settings.get("budget", 10.0)
    competitor_evaluation = False if args.no_evaluation else mode_settings.get("competitor_evaluation", True)
    
    # Interactive mode for file paths if not provided
    if not all([args.original, args.reviewer1, args.reviewer2, args.reviewer3, args.editor, args.prisma]):
        # For demo purposes, we'll use the default paths
        # In a real implementation, we'd ask the user for the paths
        original_paper_path = args.original or default_paths["original"]
        reviewer1_path = args.reviewer1 or default_paths["reviewer1"]
        reviewer2_path = args.reviewer2 or default_paths["reviewer2"]
        reviewer3_path = args.reviewer3 or default_paths["reviewer3"]
        editor_letter_path = args.editor or default_paths["editor"]
        prisma_requirements_path = args.prisma or default_paths["prisma"]
    else:
        original_paper_path = args.original
        reviewer1_path = args.reviewer1
        reviewer2_path = args.reviewer2
        reviewer3_path = args.reviewer3
        editor_letter_path = args.editor
        prisma_requirements_path = args.prisma
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create and run the tool
    tool = PaperRevisionTool(
        provider=provider,
        model_name=model,
        original_paper_path=original_paper_path,
        reviewer1_path=reviewer1_path,
        reviewer2_path=reviewer2_path,
        reviewer3_path=reviewer3_path,
        editor_letter_path=editor_letter_path,
        prisma_requirements_path=prisma_requirements_path,
        output_dir=args.output,
        operation_mode=operation_mode,
        optimize_costs=optimize_costs,
        budget=budget,
        use_cache=not args.no_cache,
        competitor_evaluation=competitor_evaluation,
        interactive=args.interactive,
        run_id=args.run_id,
    )
    
    # Run the tool and get statistics
    stats = tool.run()
    
    return 0


if __name__ == "__main__":
    main()