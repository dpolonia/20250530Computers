"""
Dependency Injection Container for the Paper Revision System.

This module provides a simple dependency injection container that manages
application dependencies and their lifecycle. It supports lazy initialization,
singleton instances, and factory methods.
"""

import inspect
import logging
from typing import Dict, Any, Callable, Type, Optional, List, Set, Union

# Configure logging
logger = logging.getLogger(__name__)


class DependencyContainer:
    """A simple dependency injection container."""
    
    def __init__(self):
        """Initialize the container."""
        self._services = {}  # type: Dict[str, Any]
        self._factories = {}  # type: Dict[str, Callable[..., Any]]
        self._singletons = set()  # type: Set[str]
        self._instances = {}  # type: Dict[str, Any]
        self._dependencies = {}  # type: Dict[str, List[str]]
    
    def register(self, name: str, service: Any, singleton: bool = True):
        """Register a service.
        
        Args:
            name: Name of the service
            service: Service class or instance
            singleton: Whether the service should be a singleton
        """
        if name in self._services:
            logger.warning(f"Service '{name}' already registered, overwriting")
            
        self._services[name] = service
        
        if singleton:
            self._singletons.add(name)
            
        logger.debug(f"Registered service '{name}', singleton={singleton}")
    
    def register_factory(self, name: str, factory: Callable[..., Any], singleton: bool = True):
        """Register a factory function for a service.
        
        Args:
            name: Name of the service
            factory: Factory function that creates the service
            singleton: Whether the service should be a singleton
        """
        if name in self._factories:
            logger.warning(f"Factory for '{name}' already registered, overwriting")
            
        self._factories[name] = factory
        
        if singleton:
            self._singletons.add(name)
            
        # Analyze factory dependencies
        self._analyze_dependencies(name, factory)
            
        logger.debug(f"Registered factory for '{name}', singleton={singleton}")
    
    def _analyze_dependencies(self, name: str, factory: Callable[..., Any]):
        """Analyze factory function to determine its dependencies.
        
        Args:
            name: Name of the service
            factory: Factory function
        """
        try:
            # Get parameter names from the function signature
            sig = inspect.signature(factory)
            param_names = [param.name for param in sig.parameters.values() 
                         if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                         or param.kind == inspect.Parameter.KEYWORD_ONLY]
            
            self._dependencies[name] = param_names
            logger.debug(f"Service '{name}' depends on: {', '.join(param_names)}")
        except Exception as e:
            logger.warning(f"Failed to analyze dependencies for '{name}': {e}")
            self._dependencies[name] = []
    
    def get(self, name: str) -> Any:
        """Get a service instance.
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If the service is not registered
        """
        # Check if instance already exists for singleton
        if name in self._singletons and name in self._instances:
            return self._instances[name]
        
        # Create the instance
        instance = self._create_instance(name)
        
        # Cache singleton instances
        if name in self._singletons:
            self._instances[name] = instance
        
        return instance
    
    def _create_instance(self, name: str) -> Any:
        """Create a service instance.
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If the service is not registered
            RecursionError: If there's a circular dependency
        """
        # Check if service exists
        if name not in self._services and name not in self._factories:
            raise ValueError(f"Service '{name}' not registered")
        
        # Use factory if available
        if name in self._factories:
            factory = self._factories[name]
            
            # Get dependencies
            dependencies = {}
            if name in self._dependencies:
                for dep_name in self._dependencies[name]:
                    if dep_name == name:
                        # Self-reference, skip to avoid circular dependency
                        continue
                    
                    try:
                        dependencies[dep_name] = self.get(dep_name)
                    except (ValueError, RecursionError) as e:
                        logger.warning(f"Failed to resolve dependency '{dep_name}' for '{name}': {e}")
                        
            # Create instance
            try:
                instance = factory(**dependencies)
                logger.debug(f"Created instance of '{name}' using factory")
                return instance
            except Exception as e:
                logger.error(f"Failed to create instance of '{name}': {e}")
                raise
        
        # Use service class
        service = self._services[name]
        
        # If it's already an instance, return it
        if not inspect.isclass(service):
            logger.debug(f"Service '{name}' is already an instance")
            return service
        
        # Create instance
        try:
            instance = service()
            logger.debug(f"Created instance of '{name}'")
            return instance
        except Exception as e:
            logger.error(f"Failed to create instance of '{name}': {e}")
            raise
    
    def has(self, name: str) -> bool:
        """Check if a service is registered.
        
        Args:
            name: Name of the service
            
        Returns:
            True if the service is registered, False otherwise
        """
        return name in self._services or name in self._factories
    
    def remove(self, name: str):
        """Remove a service.
        
        Args:
            name: Name of the service
        """
        if name in self._services:
            del self._services[name]
        
        if name in self._factories:
            del self._factories[name]
        
        if name in self._singletons:
            self._singletons.remove(name)
        
        if name in self._instances:
            del self._instances[name]
        
        if name in self._dependencies:
            del self._dependencies[name]
            
        logger.debug(f"Removed service '{name}'")
    
    def clear(self):
        """Clear all services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._instances.clear()
        self._dependencies.clear()
        logger.debug("Cleared all services")
    
    def get_all_dependencies(self) -> Dict[str, List[str]]:
        """Get all service dependencies.
        
        Returns:
            Dictionary mapping service names to lists of dependency names
        """
        return self._dependencies.copy()
    
    def get_registered_services(self) -> List[str]:
        """Get all registered service names.
        
        Returns:
            List of service names
        """
        return list(set(list(self._services.keys()) + list(self._factories.keys())))
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all dependencies can be resolved.
        
        Returns:
            List of missing dependencies
        """
        missing = []
        
        for name in self.get_registered_services():
            if name in self._dependencies:
                for dep_name in self._dependencies[name]:
                    if not self.has(dep_name) and dep_name != name:
                        missing.append(f"{name} -> {dep_name}")
        
        return missing
    
    def detect_circular_dependencies(self) -> List[str]:
        """Detect circular dependencies.
        
        Returns:
            List of circular dependency chains
        """
        circular = []
        
        def detect_cycle(name: str, path: List[str]):
            if name in path:
                # Found a cycle
                cycle_start = path.index(name)
                cycle = path[cycle_start:] + [name]
                circular.append(" -> ".join(cycle))
                return
            
            # Recursively check dependencies
            if name in self._dependencies:
                for dep_name in self._dependencies[name]:
                    if dep_name != name:  # Skip self-references
                        detect_cycle(dep_name, path + [name])
        
        # Check each service
        for name in self.get_registered_services():
            detect_cycle(name, [])
        
        return circular


# Singleton container instance
_container = DependencyContainer()

def get_container() -> DependencyContainer:
    """Get the global dependency container.
    
    Returns:
        The global dependency container
    """
    return _container