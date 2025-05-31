"""
RevisionContext: Shared state for the paper revision process.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple

from src.utils.llm_client import LLMClient
from src.config import AppConfig


class RevisionContext:
    """
    Maintains shared state for the paper revision process.

    This class serves as a container for the shared state needed by various
    components of the paper revision tool. It includes configuration settings,
    file paths, process statistics, and references to shared resources like
    the LLM client.
    """

    def __init__(
        self,
        original_paper_path: str,
        reviewer_comment_files: List[str],
        editor_letter_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        provider: str = "anthropic",
        model_name: str = "claude-3-opus-20240229",
        operation_mode: str = "finetuning",
        optimize_costs: bool = False,
        budget: float = 10.0,
        use_cache: bool = True,
        competitor_evaluation: bool = True,
        competing_evaluator: Optional[str] = None,
        interactive: bool = False,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[AppConfig] = None
    ):
        """
        Initialize the revision context with the required configuration.

        Args:
            original_paper_path: Path to the original paper PDF
            reviewer_comment_files: List of paths to reviewer comment files
            editor_letter_path: Path to the editor letter PDF
            output_dir: Directory where output files will be saved
            provider: LLM provider (anthropic, openai, google)
            model_name: Model to use
            operation_mode: Operation mode (training, finetuning, final)
            optimize_costs: Whether to optimize costs by using smaller models when possible
            budget: Maximum budget in dollars
            use_cache: Whether to use LLM response caching
            competitor_evaluation: Whether to use a competing model for evaluation
            competing_evaluator: Specific competing model to use for evaluation (provider/model)
            interactive: Whether to run in interactive mode with wait points
            run_id: Optional run ID for tracking in workflow DB
            logger: Optional logger instance
            config: Optional AppConfig instance (overrides other parameters if provided)
        """
        # Use configuration if provided
        if config is not None:
            # Store the configuration
            self.config = config
            
            # Extract configuration settings
            self.provider = config.llm.provider
            self.model_name = config.llm.model_name
            self.model = model_name.split(" (")[0] if " (" in model_name else model_name
            self.operation_mode = config.operation_mode.current_mode
            self.optimize_costs = config.budget.optimize_costs
            self.budget = config.budget.budget
            self.use_cache = config.files.use_cache
            self.competitor_evaluation = config.llm.competitor_evaluation
            self.competing_evaluator = config.llm.competing_evaluator
            self.interactive = config.interactive
            self.run_id = config.run_id
            
            # Store file paths
            self.original_paper_path = config.files.original_paper_path
            self.reviewer_comment_files = config.files.reviewer_comment_files
            self.editor_letter_path = config.files.editor_letter_path
            self.output_dir = config.files.output_dir
            
        else:
            # Store the basic configuration
            self.provider = provider
            self.model_name = model_name
            self.model = model_name.split(" (")[0] if " (" in model_name else model_name
            self.operation_mode = operation_mode
            self.optimize_costs = optimize_costs
            self.budget = budget
            self.use_cache = use_cache
            self.competitor_evaluation = competitor_evaluation
            self.competing_evaluator = competing_evaluator
            self.interactive = interactive
            self.run_id = run_id
            
            # Store file paths
            self.original_paper_path = original_paper_path
            self.reviewer_comment_files = reviewer_comment_files
            self.editor_letter_path = editor_letter_path
            self.output_dir = output_dir
            
            # Set config to None
            self.config = None

        # Derived paths
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.output_dir, f"revision_log_{self.timestamp}.txt")
        
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(self.original_paper_path))[0]
        # Get the directory containing the original paper
        original_dir = os.path.dirname(self.original_paper_path)
        
        # Look for a docx file with the same name
        self.original_paper_docx = os.path.join(original_dir, f"{original_filename}.docx")
        if not os.path.exists(self.original_paper_docx):
            # If not found, it will be created later
            self.original_paper_docx = os.path.join(self.output_dir, f"{original_filename}.docx")
            
        # Look for a BibTeX file in the same directory
        bib_files = [f for f in os.listdir(original_dir) if f.endswith('.bib')]
        if bib_files:
            self.bib_path = os.path.join(original_dir, bib_files[0])
        else:
            # If not found, it will need to be created later
            self.bib_path = os.path.join(self.output_dir, f"{original_filename}.bib")

        # Setup the tracking variables
        self.llm_client: Optional[LLMClient] = None
        self.logger: Optional[logging.Logger] = logger
        
        # Process statistics
        self.process_statistics: Dict[str, Any] = {
            "start_time": datetime.now(),
            "provider": self.provider,
            "model": self.model_name,
            "operation_mode": self.operation_mode,
            "optimize_costs": self.optimize_costs,
            "initial_budget": self.budget,
            "remaining_budget": self.budget,
            "requests": 0,
            "tokens_used": 0,
            "total_cost": 0.0,
            "files_processed": 0,
            "files_created": 0,
            "steps_completed": 0,
            "step_count": 10,  # Default number of steps
        }
        
        # Analysis storage
        self.paper_analysis: Optional[Dict[str, Any]] = None
        self.reviewer_comments: Optional[List[Dict[str, Any]]] = None
        self.editor_requirements: Optional[Dict[str, Any]] = None
        
        # Token usage settings
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # API settings
        self.api = None
        self.api_key = None
        self.verify_models = True

        # Additional metadata
        self.paper_title = None
        self.paper_authors = []

    def setup_llm_client(self, verify: bool = True) -> LLMClient:
        """
        Set up the LLM client for this context.

        Args:
            verify: Whether to verify the model existence

        Returns:
            The configured LLM client
        """
        from src.utils.llm_client import get_llm_client
        
        self.llm_client = get_llm_client(
            provider=self.provider,
            model=self.model,
            verify=verify
        )
        
        # If we have a run ID and the client supports it, set it for tracking
        if self.run_id and hasattr(self.llm_client, 'set_run_id'):
            self.llm_client.set_run_id(self.run_id)
            
        return self.llm_client
    
    def setup_logger(self) -> logging.Logger:
        """
        Set up the logger for this context.

        Returns:
            The configured logger
        """
        # Create logger
        logger = logging.getLogger(f"revision_{self.timestamp}")
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
    
    def update_budget(self, tokens: int, cost: float) -> bool:
        """
        Update the budget with new token usage and cost.

        Args:
            tokens: Number of tokens used
            cost: Cost incurred

        Returns:
            True if the budget is still available, False if it has been exceeded
        """
        self.total_tokens_used += tokens
        self.total_cost += cost
        
        # Update process statistics
        self.process_statistics["tokens_used"] = self.total_tokens_used
        self.process_statistics["total_cost"] = self.total_cost
        self.process_statistics["remaining_budget"] = self.budget - self.total_cost
        self.process_statistics["requests"] += 1
        
        # Check if we're over budget
        return self.total_cost <= self.budget
    
    def check_budget(self) -> bool:
        """
        Check if we're still within budget.

        Returns:
            True if within budget, False otherwise
        """
        return self.total_cost <= self.budget
    
    def log_info(self, message: str) -> None:
        """Log an informational message."""
        if self.logger:
            self.logger.info(message)
            
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        if self.logger:
            self.logger.warning(message)
            
    def log_error(self, message: str) -> None:
        """Log an error message."""
        if self.logger:
            self.logger.error(message)
            
    def log_success(self, message: str) -> None:
        """Log a success message."""
        if self.logger:
            self.logger.info(f"SUCCESS: {message}")
            
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        if self.logger:
            self.logger.debug(message)
    
    def get_output_path(self, filename: str) -> str:
        """
        Get a full path for an output file.

        Args:
            filename: The filename to use

        Returns:
            Full path in the output directory
        """
        return os.path.join(self.output_dir, filename)
    
    def complete_step(self) -> None:
        """Mark a step as completed in the process statistics."""
        self.process_statistics["steps_completed"] += 1
    
    @classmethod
    def from_config(cls, config: AppConfig, logger: Optional[logging.Logger] = None) -> 'RevisionContext':
        """
        Create a RevisionContext from an AppConfig.
        
        Args:
            config: Application configuration
            logger: Optional logger instance
            
        Returns:
            RevisionContext instance
        """
        return cls(
            original_paper_path=config.files.original_paper_path,
            reviewer_comment_files=config.files.reviewer_comment_files,
            editor_letter_path=config.files.editor_letter_path,
            output_dir=config.files.output_dir,
            provider=config.llm.provider,
            model_name=config.llm.model_name,
            operation_mode=config.operation_mode.current_mode,
            optimize_costs=config.budget.optimize_costs,
            budget=config.budget.budget,
            use_cache=config.files.use_cache,
            competitor_evaluation=config.llm.competitor_evaluation,
            competing_evaluator=config.llm.competing_evaluator,
            interactive=config.interactive,
            run_id=config.run_id,
            logger=logger,
            config=config
        )