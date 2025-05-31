"""
Interface definitions for budget management components.

This module defines abstract base classes that serve as interfaces for the
budget management components, ensuring consistent design patterns across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from src.core.context import RevisionContext


class BudgetManagerInterface(ABC):
    """
    Base interface for budget management components.
    
    Budget managers are responsible for tracking and managing token usage and costs
    during the paper revision process.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the budget manager with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
    
    @abstractmethod
    def update(self, tokens: int, cost: float) -> bool:
        """
        Update the budget with new token usage and cost.
        
        Args:
            tokens: Number of tokens used
            cost: Cost incurred
            
        Returns:
            True if the budget is still available, False if it has been exceeded
        """
        pass
    
    @abstractmethod
    def check(self) -> bool:
        """
        Check if we're still within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_operation(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        provider: str, 
        model: str
    ) -> Tuple[float, bool]:
        """
        Estimate the cost of an operation and check if it's within budget.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: The LLM provider (anthropic, openai, google)
            model: The model name
            
        Returns:
            Tuple of (estimated cost, whether it's within budget)
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get budget statistics.
        
        Returns:
            Dictionary with budget statistics
        """
        pass
    
    @abstractmethod
    def generate_report(self) -> str:
        """
        Generate a cost report.
        
        Returns:
            String containing the cost report
        """
        pass