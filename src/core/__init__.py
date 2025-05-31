"""Core module for the paper revision tool."""

from .context import RevisionContext
from .paper_revision import PaperRevisionTool, choose_model, choose_operation_mode
from .interactive import interactive_wait
from .constants import OPERATION_MODES, TASK_TYPES
from .json_utils import extract_json_from_text, parse_json_safely
from .factory import (
    create_paper_analyzer,
    create_reviewer_analyzer,
    create_solution_generator,
    create_changes_document_generator,
    create_revised_paper_generator,
    create_assessment_generator,
    create_editor_letter_generator,
    create_reference_manager,
    create_quality_evaluator,
    create_budget_manager,
    register_component
)
from .patterns import (
    Component,
    Strategy,
    StrategyContext,
    Observer,
    Subject,
    Singleton,
    memoize
)

__all__ = [
    # Core classes
    'RevisionContext',
    'PaperRevisionTool',
    
    # Functions
    'choose_model',
    'choose_operation_mode',
    'interactive_wait',
    'extract_json_from_text',
    'parse_json_safely',
    
    # Constants
    'OPERATION_MODES',
    'TASK_TYPES',
    
    # Factory functions
    'create_paper_analyzer',
    'create_reviewer_analyzer',
    'create_solution_generator',
    'create_changes_document_generator',
    'create_revised_paper_generator',
    'create_assessment_generator',
    'create_editor_letter_generator',
    'create_reference_manager',
    'create_quality_evaluator',
    'create_budget_manager',
    'register_component',
    
    # Design patterns
    'Component',
    'Strategy',
    'StrategyContext',
    'Observer',
    'Subject',
    'Singleton',
    'memoize'
]
