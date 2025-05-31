"""
Common design patterns for the paper revision tool.

This module provides common design patterns that can be used across the codebase
to ensure consistent implementation and behavior.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Generic, TypeVar, Optional, Callable

from src.core.context import RevisionContext


# Type variable for the strategy pattern
T = TypeVar('T')


class Component(ABC):
    """
    Base class for all components in the paper revision tool.
    
    Components are responsible for specific tasks and have access to the shared
    revision context. This base class provides common functionality and enforces
    consistent design patterns.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the component with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(self.__class__.__name__)
    
    def _log_info(self, message: str) -> None:
        """Log an informational message."""
        if self.logger:
            self.logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log a warning message."""
        if self.logger:
            self.logger.warning(message)
    
    def _log_error(self, message: str) -> None:
        """Log an error message."""
        if self.logger:
            self.logger.error(message)
    
    def _log_success(self, message: str) -> None:
        """Log a success message."""
        if self.logger:
            self.logger.info(f"SUCCESS: {message}")
    
    def _log_debug(self, message: str) -> None:
        """Log a debug message."""
        if self.logger:
            self.logger.debug(message)
    
    def _get_completion(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """
        Get a completion from the LLM with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If there was an error getting the completion
        """
        try:
            # Ensure LLM client is initialized
            if not self.context.llm_client:
                self.context.setup_llm_client()
                
            # Get completion from LLM
            response = self.context.llm_client.get_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
            
            # Update budget
            tokens_used = self.context.llm_client.total_tokens_used
            cost = self.context.llm_client.total_cost
            self.context.update_budget(tokens_used, cost)
            
            return response
            
        except Exception as e:
            self._log_error(f"Error getting completion: {e}")
            raise


class Strategy(Generic[T], ABC):
    """
    Base class for the Strategy pattern.
    
    Strategies encapsulate algorithms that can be interchanged at runtime.
    This allows for flexible and configurable behavior.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        """
        Execute the strategy.
        
        Returns:
            The result of the strategy execution
        """
        pass


class StrategyContext(Generic[T]):
    """
    Context class for the Strategy pattern.
    
    This class maintains a reference to a strategy object and delegates
    the execution to the strategy.
    """
    
    def __init__(self, strategy: Strategy[T]):
        """
        Initialize the context with a strategy.
        
        Args:
            strategy: The strategy to use
        """
        self._strategy = strategy
    
    @property
    def strategy(self) -> Strategy[T]:
        """Get the current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: Strategy[T]) -> None:
        """
        Set the strategy.
        
        Args:
            strategy: The strategy to use
        """
        self._strategy = strategy
    
    def execute(self, *args, **kwargs) -> T:
        """
        Execute the strategy.
        
        Returns:
            The result of the strategy execution
        """
        return self._strategy.execute(*args, **kwargs)


class Observer(ABC):
    """
    Base class for the Observer pattern.
    
    Observers receive notifications from subjects they are observing.
    """
    
    @abstractmethod
    def update(self, subject: 'Subject', *args, **kwargs) -> None:
        """
        Update the observer with new information from the subject.
        
        Args:
            subject: The subject that triggered the update
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        pass


class Subject(ABC):
    """
    Base class for the Subject pattern.
    
    Subjects maintain a list of observers and notify them of changes.
    """
    
    def __init__(self):
        """Initialize the subject with an empty list of observers."""
        self._observers: list[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the subject.
        
        Args:
            observer: The observer to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the subject.
        
        Args:
            observer: The observer to detach
        """
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
    
    def notify(self, *args, **kwargs) -> None:
        """
        Notify all observers of a change.
        
        Args:
            *args: Additional arguments to pass to observers
            **kwargs: Additional keyword arguments to pass to observers
        """
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


class Singleton:
    """
    Base class for the Singleton pattern.
    
    Singletons ensure that a class has only one instance and provide a global
    point of access to that instance.
    """
    
    _instances: Dict[type, Any] = {}
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance or return the existing one.
        
        Returns:
            The singleton instance
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]


def memoize(func: Callable) -> Callable:
    """
    Decorator for memoization.
    
    Memoization caches the results of function calls to avoid redundant
    computation.
    
    Args:
        func: The function to memoize
        
    Returns:
        The memoized function
    """
    cache: Dict[tuple, Any] = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper