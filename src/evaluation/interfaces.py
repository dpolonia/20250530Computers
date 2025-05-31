"""
Interface definitions for evaluation components.

This module defines abstract base classes that serve as interfaces for the
evaluation components, ensuring consistent design patterns across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from src.core.context import RevisionContext


class EvaluatorInterface(ABC):
    """
    Base interface for evaluation components.
    
    Evaluators are responsible for assessing the quality of model responses
    and providing feedback for improvement.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the evaluator with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform evaluation and return results.
        
        Returns:
            Dictionary with evaluation results
        """
        pass


class QualityEvaluatorInterface(EvaluatorInterface):
    """Interface for quality evaluation components."""
    
    @abstractmethod
    def evaluate_response_quality(
        self, 
        prompt: str, 
        response: str, 
        task_type: str = "general",
        use_competitor: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a model response.
        
        Args:
            prompt: The original prompt sent to the model
            response: The model's response
            task_type: The type of task (analysis, generation, editing, etc.)
            use_competitor: Whether to use a competing model for evaluation
            
        Returns:
            Dictionary with quality metrics and feedback
        """
        pass
    
    @abstractmethod
    def get_competing_model(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get a competing model for evaluation.
        
        Returns:
            Tuple of (provider, model_name) for the competing model
        """
        pass