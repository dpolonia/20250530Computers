"""
Factory for creating components with consistent interfaces.

This module provides factory functions for creating components that implement
the interfaces defined in the various modules, ensuring consistent design patterns
across the codebase.
"""

from typing import Dict, Any, Optional, Type

from src.core.context import RevisionContext

# Analysis components
from src.analysis.interfaces import (
    PaperAnalyzerInterface,
    ReviewerAnalyzerInterface,
    SolutionGeneratorInterface
)
from src.analysis.paper_analyzer import PaperAnalyzer
from src.analysis.reviewer_analyzer import ReviewerAnalyzer
from src.analysis.solution_generator import SolutionGenerator

# Document components
from src.document.interfaces import (
    ChangesDocumentGeneratorInterface,
    RevisedPaperGeneratorInterface,
    AssessmentGeneratorInterface,
    EditorLetterGeneratorInterface
)
from src.document.changes_document import ChangesDocumentGenerator
from src.document.revised_paper import RevisedPaperGenerator
from src.document.assessment import AssessmentGenerator
from src.document.editor_letter import EditorLetterGenerator

# Reference components
from src.references.interfaces import ReferenceManagerInterface
from src.references.reference_manager import ReferenceManager

# Evaluation components
from src.evaluation.interfaces import QualityEvaluatorInterface
from src.evaluation.quality_evaluator import QualityEvaluator

# Budget components
from src.budget.interfaces import BudgetManagerInterface
from src.budget.budget_manager import BudgetManager


# Component registry for dependency injection
_component_registry: Dict[Type, Type] = {
    # Analysis components
    PaperAnalyzerInterface: PaperAnalyzer,
    ReviewerAnalyzerInterface: ReviewerAnalyzer,
    SolutionGeneratorInterface: SolutionGenerator,
    
    # Document components
    ChangesDocumentGeneratorInterface: ChangesDocumentGenerator,
    RevisedPaperGeneratorInterface: RevisedPaperGenerator,
    AssessmentGeneratorInterface: AssessmentGenerator,
    EditorLetterGeneratorInterface: EditorLetterGenerator,
    
    # Reference components
    ReferenceManagerInterface: ReferenceManager,
    
    # Evaluation components
    QualityEvaluatorInterface: QualityEvaluator,
    
    # Budget components
    BudgetManagerInterface: BudgetManager
}


def register_component(interface_class: Type, implementation_class: Type) -> None:
    """
    Register a component implementation for an interface.
    
    Args:
        interface_class: The interface class
        implementation_class: The implementation class
    """
    _component_registry[interface_class] = implementation_class


def create_component(interface_class: Type, context: RevisionContext, **kwargs) -> Any:
    """
    Create a component instance that implements the given interface.
    
    Args:
        interface_class: The interface class
        context: The revision context
        **kwargs: Additional arguments to pass to the component constructor
        
    Returns:
        An instance of the component
        
    Raises:
        ValueError: If no implementation is registered for the interface
    """
    if interface_class not in _component_registry:
        raise ValueError(f"No implementation registered for interface {interface_class.__name__}")
    
    implementation_class = _component_registry[interface_class]
    return implementation_class(context, **kwargs)


# Factory functions for specific components

def create_paper_analyzer(context: RevisionContext) -> PaperAnalyzerInterface:
    """Create a paper analyzer component."""
    return create_component(PaperAnalyzerInterface, context)


def create_reviewer_analyzer(context: RevisionContext) -> ReviewerAnalyzerInterface:
    """Create a reviewer analyzer component."""
    return create_component(ReviewerAnalyzerInterface, context)


def create_solution_generator(context: RevisionContext) -> SolutionGeneratorInterface:
    """Create a solution generator component."""
    return create_component(SolutionGeneratorInterface, context)


def create_changes_document_generator(context: RevisionContext) -> ChangesDocumentGeneratorInterface:
    """Create a changes document generator component."""
    return create_component(ChangesDocumentGeneratorInterface, context)


def create_revised_paper_generator(context: RevisionContext) -> RevisedPaperGeneratorInterface:
    """Create a revised paper generator component."""
    return create_component(RevisedPaperGeneratorInterface, context)


def create_assessment_generator(context: RevisionContext) -> AssessmentGeneratorInterface:
    """Create an assessment generator component."""
    return create_component(AssessmentGeneratorInterface, context)


def create_editor_letter_generator(context: RevisionContext) -> EditorLetterGeneratorInterface:
    """Create an editor letter generator component."""
    return create_component(EditorLetterGeneratorInterface, context)


def create_reference_manager(context: RevisionContext) -> ReferenceManagerInterface:
    """Create a reference manager component."""
    return create_component(ReferenceManagerInterface, context)


def create_quality_evaluator(context: RevisionContext) -> QualityEvaluatorInterface:
    """Create a quality evaluator component."""
    return create_component(QualityEvaluatorInterface, context)


def create_budget_manager(context: RevisionContext, **kwargs) -> BudgetManagerInterface:
    """Create a budget manager component."""
    return create_component(BudgetManagerInterface, context, **kwargs)