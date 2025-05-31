"""
Service factory implementation.

This module provides a factory for creating service instances with proper dependency
injection.
"""

import logging
from typing import Dict, Any, Type, Optional

from src.core.context import RevisionContext
from src.services.interfaces import (
    ServiceInterface,
    PaperServiceInterface,
    ReviewerServiceInterface,
    SolutionServiceInterface,
    DocumentServiceInterface,
    ReferenceServiceInterface,
    LLMServiceInterface
)
from src.services.paper_service import PaperService
from src.services.reviewer_service import ReviewerService
from src.services.solution_service import SolutionService
from src.services.document_service import DocumentService
from src.services.reference_service import ReferenceService
from src.services.llm_service import LLMService


class ServiceFactory:
    """
    Factory for creating service instances.
    
    This factory is responsible for creating service instances with proper
    dependency injection, ensuring that all services share the same context.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the service factory.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self._instances: Dict[Type[ServiceInterface], ServiceInterface] = {}
    
    def get_paper_service(self) -> PaperServiceInterface:
        """
        Get a paper service instance.
        
        Returns:
            PaperService instance
        """
        return self._get_or_create(PaperServiceInterface, PaperService)
    
    def get_reviewer_service(self) -> ReviewerServiceInterface:
        """
        Get a reviewer service instance.
        
        Returns:
            ReviewerService instance
        """
        return self._get_or_create(ReviewerServiceInterface, ReviewerService)
    
    def get_solution_service(self) -> SolutionServiceInterface:
        """
        Get a solution service instance.
        
        Returns:
            SolutionService instance
        """
        return self._get_or_create(SolutionServiceInterface, SolutionService)
    
    def get_document_service(self) -> DocumentServiceInterface:
        """
        Get a document service instance.
        
        Returns:
            DocumentService instance
        """
        return self._get_or_create(DocumentServiceInterface, DocumentService)
    
    def get_reference_service(self) -> ReferenceServiceInterface:
        """
        Get a reference service instance.
        
        Returns:
            ReferenceService instance
        """
        return self._get_or_create(ReferenceServiceInterface, ReferenceService)
    
    def get_llm_service(self) -> LLMServiceInterface:
        """
        Get an LLM service instance.
        
        Returns:
            LLMService instance
        """
        return self._get_or_create(LLMServiceInterface, LLMService)
    
    def _get_or_create(
        self, 
        interface_class: Type[ServiceInterface], 
        implementation_class: Type[ServiceInterface]
    ) -> ServiceInterface:
        """
        Get an existing service instance or create a new one.
        
        Args:
            interface_class: The interface class
            implementation_class: The implementation class
            
        Returns:
            Service instance
        """
        if interface_class not in self._instances:
            self._instances[interface_class] = implementation_class(self.context)
            self.logger.debug(f"Created new instance of {implementation_class.__name__}")
        
        return self._instances[interface_class]