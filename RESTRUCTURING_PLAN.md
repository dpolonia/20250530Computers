# Paper Revision Tool Restructuring Plan

## Problem Statement
The current codebase has a large monolithic file (`paper_revision.py` at ~5000 lines) that exceeds analysis limits for AI assistants and makes code maintenance challenging.

## Proposed Solution
Restructure the codebase into smaller, more focused modules while preserving functionality.

## New Directory Structure

```
src/
├── core/
│   ├── __init__.py
│   ├── paper_revision.py        # Main entry point and orchestrator
│   ├── constants.py             # Constants and shared configurations
│   └── interactive.py           # Interactive mode functionality
│
├── document/
│   ├── __init__.py
│   ├── changes_document.py      # Creates document detailing changes
│   ├── editor_letter.py         # Creates letter to editor
│   ├── assessment.py            # Creates assessment document
│   └── revised_paper.py         # Creates the revised paper
│
├── analysis/
│   ├── __init__.py
│   ├── paper_analyzer.py        # Analyzes original paper
│   ├── reviewer_analyzer.py     # Analyzes reviewer comments
│   └── solution_generator.py    # Generates solutions to issues
│
├── budget/
│   ├── __init__.py
│   ├── budget_manager.py        # Manages token and cost budgets
│   └── statistics.py            # Tracks and reports process statistics
│
├── evaluation/
│   ├── __init__.py
│   ├── quality_evaluator.py     # Evaluates response quality
│   └── cross_model.py           # Cross-model evaluation
│
├── references/
│   ├── __init__.py
│   └── reference_manager.py     # Validates and updates references
│
├── utils/                       # Existing utilities
│   ├── __init__.py
│   ├── document_processor.py
│   ├── llm_client.py
│   ├── pdf_processor.py
│   └── reference_validator.py
│
└── models/                      # Existing models
    ├── __init__.py
    ├── anthropic_models.py
    ├── google_models.py
    └── openai_models.py
```

## Refactoring Approach

1. **Extract Core Functionality**:
   - Maintain the `PaperRevisionTool` class as the main entry point but make it smaller
   - Move specialized functionality to appropriate modules
   - Use dependency injection for services

2. **Share Common State**:
   - Create a `RevisionContext` class to hold shared state
   - Pass this context to specialized classes 

3. **Define Clear Interfaces**:
   - Each module should have a clear public API
   - Use proper typing and docstrings

4. **Handle Dependencies**:
   - Use factory methods to create dependencies
   - Consider a simple dependency injection approach

## Implementation Steps

1. Create the directory structure
2. Create stub files with proper imports and class definitions
3. Extract functionality from the monolithic file into appropriate modules
4. Update imports and references
5. Add tests for each module
6. Refactor main entry point (`paper_revision.py`) to use the new modules

## Backwards Compatibility

The external interface of the tool will remain unchanged. Users should not notice any difference in how they interact with the tool.

## Testing Strategy

1. Create tests for each new module
2. Compare output of new implementation with the original 
3. Run full integration tests to ensure overall behavior is preserved

## Timeline

1. Setup directory structure and stub files (1 day)
2. Extract core and document modules (2 days)
3. Extract analysis and references modules (2 days)
4. Extract budget and evaluation modules (1 day)
5. Update main entry point (1 day)
6. Testing and fixes (2 days)

Total estimated time: 9 days