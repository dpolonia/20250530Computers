"""
Budget management for the paper revision tool.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime

from src.core.constants import BUDGET_THRESHOLDS


class BudgetManager:
    """
    Manages token and cost budgets for LLM operations.
    """
    
    def __init__(
        self, 
        budget: float, 
        logger: Optional[logging.Logger] = None,
        statistics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the budget manager.
        
        Args:
            budget: Maximum budget in dollars
            logger: Optional logger for budget messages
            statistics: Optional dictionary to store budget statistics
        """
        self.initial_budget = budget
        self.remaining_budget = budget
        self.logger = logger
        self.statistics = statistics or {}
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Initialize statistics if not provided
        if not statistics:
            self.statistics = {
                "start_time": datetime.now(),
                "initial_budget": budget,
                "remaining_budget": budget,
                "requests": 0,
                "tokens_used": 0,
                "total_cost": 0.0,
            }
    
    def update(self, tokens: int, cost: float) -> bool:
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
        self.remaining_budget = self.initial_budget - self.total_cost
        
        # Update statistics
        self.statistics["tokens_used"] = self.total_tokens_used
        self.statistics["total_cost"] = self.total_cost
        self.statistics["remaining_budget"] = self.remaining_budget
        self.statistics["requests"] = self.statistics.get("requests", 0) + 1
        
        # Log budget status if logger is available
        if self.logger:
            # Log based on threshold status
            usage_ratio = self.total_cost / self.initial_budget
            if usage_ratio >= BUDGET_THRESHOLDS["critical"]:
                self.logger.warning(
                    f"BUDGET CRITICAL: ${self.total_cost:.2f} used out of ${self.initial_budget:.2f} "
                    f"({usage_ratio*100:.1f}%). ${self.remaining_budget:.2f} remaining."
                )
            elif usage_ratio >= BUDGET_THRESHOLDS["warning"]:
                self.logger.warning(
                    f"BUDGET WARNING: ${self.total_cost:.2f} used out of ${self.initial_budget:.2f} "
                    f"({usage_ratio*100:.1f}%). ${self.remaining_budget:.2f} remaining."
                )
            else:
                self.logger.debug(
                    f"Budget update: ${self.total_cost:.2f} used, ${self.remaining_budget:.2f} remaining."
                )
        
        # Return whether we're still within budget
        return self.remaining_budget >= 0
    
    def check(self) -> bool:
        """
        Check if we're still within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        return self.remaining_budget >= 0
    
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
        # Get rate information for the model
        from src.utils.llm_client import get_model_rates
        rates = get_model_rates(provider, model)
        
        if not rates:
            if self.logger:
                self.logger.warning(f"Could not find rate information for {provider}/{model}")
            # Default to a high estimate if rates unknown
            estimated_cost = (input_tokens + output_tokens) * 0.01 / 1000
        else:
            input_rate = rates.get("price_per_1k_input", 0.0)
            output_rate = rates.get("price_per_1k_output", 0.0)
            
            estimated_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1000
        
        # Check if the operation is within budget
        within_budget = (self.total_cost + estimated_cost) <= self.initial_budget
        
        return estimated_cost, within_budget
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get budget statistics.
        
        Returns:
            Dictionary with budget statistics
        """
        # Update duration
        duration = (datetime.now() - self.statistics["start_time"]).total_seconds()
        self.statistics["total_duration"] = duration
        
        # Calculate usage percentage
        if self.initial_budget > 0:
            self.statistics["budget_usage_percentage"] = (self.total_cost / self.initial_budget) * 100
        else:
            self.statistics["budget_usage_percentage"] = 0
        
        return self.statistics
    
    def generate_report(self) -> str:
        """
        Generate a cost report.
        
        Returns:
            String containing the cost report
        """
        stats = self.get_statistics()
        
        # Calculate duration in hours, minutes, seconds
        duration = stats.get("total_duration", 0)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        report = [
            "=" * 50,
            "PAPER REVISION COST REPORT",
            "=" * 50,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
            "-" * 50,
            f"Total API requests: {stats.get('requests', 0)}",
            f"Total tokens used: {stats.get('tokens_used', 0):,}",
            f"Total cost: ${stats.get('total_cost', 0.0):.4f}",
            f"Initial budget: ${stats.get('initial_budget', 0.0):.2f}",
            f"Remaining budget: ${stats.get('remaining_budget', 0.0):.2f}",
            f"Budget usage: {stats.get('budget_usage_percentage', 0):.1f}%",
            "=" * 50,
        ]
        
        return "\n".join(report)