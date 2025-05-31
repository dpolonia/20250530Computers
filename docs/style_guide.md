# Documentation Style Guide

This document outlines the documentation standards for the Paper Revision Tool project. Consistent documentation makes the codebase easier to understand, maintain, and extend.

## Docstring Format

We use Google-style docstrings throughout the codebase. This format is readable in both plain text and when processed by documentation generators.

### Module-Level Docstrings

Every module must have a module-level docstring that explains:

1. The purpose of the module
2. Key components defined in the module
3. How the module fits into the overall architecture

Example:
```python
"""
Document processing utilities for the paper revision tool.

This module provides classes and functions for parsing, analyzing, and
manipulating academic documents in various formats (PDF, DOCX, etc.).
It is used by the document service layer to extract structured information
from papers and reviewer comments.
"""
```

### Class Docstrings

Class docstrings must explain:

1. The purpose and responsibility of the class
2. How it fits into the overall architecture
3. Any design patterns it implements
4. Usage examples for complex classes

Example:
```python
class DocumentProcessor:
    """
    Processes document files to extract structured information.
    
    This class is responsible for parsing document files (PDF, DOCX, etc.),
    extracting their content, and organizing it into a structured format.
    It acts as a facade over various document format adapters.
    
    Usage:
        processor = DocumentProcessor(context)
        document = processor.process("path/to/document.pdf")
    """
```

### Method/Function Docstrings

Method and function docstrings must include:

1. A brief description of what the method does
2. Parameters (name, type, description)
3. Return value (type, description)
4. Exceptions that may be raised
5. Examples for complex methods

Example:
```python
def extract_sections(document_text: str, min_length: int = 100) -> Dict[str, str]:
    """
    Extract sections from document text.
    
    Args:
        document_text: The text of the document to process
        min_length: Minimum character length for a section to be included
        
    Returns:
        Dictionary mapping section names to their content
        
    Raises:
        ValueError: If document_text is empty or not a string
        
    Example:
        >>> text = "Introduction\\nThis is content...\\nMethods\\nMore content..."
        >>> sections = extract_sections(text)
        >>> print(sections.keys())
        ['Introduction', 'Methods']
    """
```

### Property Docstrings

Properties should be documented with:

1. A brief description
2. Return type and description
3. Any exceptions that may be raised

Example:
```python
@property
def title(self) -> str:
    """
    Get the document title.
    
    Returns:
        The title of the document
        
    Raises:
        AttributeError: If title extraction failed
    """
```

### Interface Docstrings

Interfaces should document:

1. The purpose of the interface
2. Implementation requirements
3. Expected behavior of implementations

Example:
```python
class DocumentServiceInterface:
    """
    Interface for document services.
    
    This interface defines the contract for services that create
    and manipulate documents related to the paper revision process.
    
    Implementations must handle different document formats and ensure
    that all created documents follow the required structure.
    """
```

## Documentation Consistency

### Types

Use type hints consistently:

```python
def process_document(path: str, validate: bool = True) -> Dict[str, Any]:
    """Process a document."""
```

Use descriptive type aliases for complex types:

```python
# At module level
SectionDict = Dict[str, str]
ReviewerFeedback = List[Dict[str, Any]]

def extract_feedback(comments: List[str]) -> ReviewerFeedback:
    """Extract feedback from comments."""
```

### Optional Parameters

Document optional parameters with their default values:

```python
def analyze_paper(path: str, depth: int = 2, extract_references: bool = True) -> Dict[str, Any]:
    """
    Analyze a paper document.
    
    Args:
        path: Path to the paper file
        depth: Analysis depth level (default: 2)
        extract_references: Whether to extract references (default: True)
    """
```

### Return Values

Always document return values with types and descriptions:

```python
def get_paper_sections() -> Dict[str, str]:
    """
    Get paper sections.
    
    Returns:
        Dictionary mapping section names to their content
    """
```

### Exceptions

Document all exceptions that might be raised:

```python
def validate_references(references: List[str]) -> List[str]:
    """
    Validate references.
    
    Args:
        references: List of reference strings to validate
        
    Returns:
        List of validated and normalized references
        
    Raises:
        ValueError: If any reference has invalid format
        ConnectionError: If the validation service is unavailable
    """
```

## Code Examples

Include code examples for complex functions or classes:

```python
def parse_bibtex(bibtex_str: str) -> List[Dict[str, str]]:
    """
    Parse BibTeX string into structured references.
    
    Args:
        bibtex_str: BibTeX string to parse
        
    Returns:
        List of dictionaries containing parsed reference fields
        
    Example:
        >>> bibtex = '@article{smith2020, author="Smith, J.", title="Example"}'
        >>> refs = parse_bibtex(bibtex)
        >>> refs[0]['title']
        'Example'
    """
```

## Architectural Documentation

### Module-Level Architecture

Each package (`src/core`, `src/services`, etc.) should have a `README.md` file that describes:

1. The purpose of the package
2. Key components and their relationships
3. How the package interacts with other packages
4. Usage examples

### Interface Documentation

Interfaces should document:

1. The contract they define
2. Extension points
3. Implementation requirements
4. Lifecycle management (if applicable)

### Design Patterns

When a design pattern is used, document:

1. The pattern name
2. Why it was chosen
3. How it's implemented
4. Any variations from the standard pattern

Example:
```python
class ServiceFactory:
    """
    Factory for creating service instances.
    
    This class implements the Factory pattern to create service instances
    based on the current context. It ensures that services are properly
    initialized with their dependencies and context.
    
    The factory caches service instances to avoid redundant instantiation.
    """
```

## File Headers

Each file should include a standardized header:

```python
"""
<module_name>.py: <brief description>

This module <detailed description>.

Copyright (c) 2025 Paper Revision Tool
"""
```

## TODOs and Technical Debt

Document technical debt with TODO comments that include:

1. A description of what needs to be done
2. Why it's important
3. A ticket or issue reference when applicable

Example:
```python
# TODO: Implement caching for API responses to reduce token usage
# This would significantly reduce costs for repeated operations
# Issue: #42
```

## Documentation Files

Documentation files should:

1. Use Markdown format
2. Include a descriptive title
3. Have a clear structure with headings
4. Include examples when appropriate
5. Cross-reference related documentation

## Version Information

Include version information in API documentation:

```python
def deprecated_function():
    """
    This function is deprecated since v1.2.0 and will be removed in v2.0.0.
    Use new_function() instead.
    """
```

## External References

When referring to external resources, include:

1. Title of the resource
2. URL (if applicable)
3. Version or date of the resource
4. Brief description of relevance

## Maintaining Documentation

1. Update documentation when code changes
2. Review documentation during code reviews
3. Test examples in documentation to ensure they work
4. Keep architectural documentation in sync with implementation