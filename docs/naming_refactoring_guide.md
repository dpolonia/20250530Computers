# Naming Refactoring Guide

This guide outlines the process for systematically refactoring naming conventions in the Paper Revision Tool codebase.

## Approach

To minimize disruption while improving naming consistency, we'll follow a phased approach:

1. **Fix ambiguous names**: Replace vague function names first
2. **Standardize parameter names**: Ensure consistency across similar methods
3. **Fix casing inconsistencies**: Convert all names to the proper convention
4. **Fix imports and references**: Update all references to renamed elements

## Step 1: Identify Naming Issues

Use the naming checker script to identify naming issues:

```bash
python scripts/naming_checker.py --directory src
```

The script will produce a report of naming convention issues. Focus on:

1. Ambiguous function/method names
2. Inconsistent parameter names
3. Casing inconsistencies (camelCase vs snake_case)
4. File name inconsistencies

## Step 2: Prioritize Changes

Focus on the most critical issues first:

1. **Critical Priority**:
   - Ambiguous public API names (methods that are part of the main interface)
   - Inconsistent parameter names in the most commonly used methods
   - Mixed casing styles within the same file or class

2. **Medium Priority**:
   - Internal method names that are ambiguous
   - Parameter names that use non-standard terms
   - Minor casing inconsistencies in implementation details

3. **Low Priority**:
   - Private method naming (as long as they're consistent within a class)
   - Variable names deep in implementation details
   - File and module names (require coordinated changes)

## Step 3: Rename Functions and Methods

When renaming functions and methods:

1. Update the name in the function/method definition
2. Update all direct references to the function/method
3. Update any imports that reference the function/method
4. Update any documentation that references the function/method

Example approach:

```python
# Before
def process(data):
    # Implementation...

# After
def process_document_content(data):
    # Implementation...
```

## Step 4: Standardize Parameter Names

When standardizing parameter names:

1. Update the parameter name in the function signature
2. Update all references to the parameter within the function body
3. Update docstring parameter descriptions

Use these standard parameter names:

| Concept | Standard Name | Non-Standard Names to Replace |
|---------|--------------|-------------------------------|
| File path | `file_path` | `path`, `file`, `filepath`, `filename` |
| Output path | `output_path` | `output`, `output_file`, `outpath` |
| API key | `api_key` | `key`, `apikey`, `api_token` |
| Model name | `model_name` | `model`, `model_id`, `llm` |

Example approach:

```python
# Before
def extract_text(file, format_type="auto"):
    # Implementation using 'file'...

# After
def extract_text(file_path, format_type="auto"):
    # Implementation using 'file_path'...
```

## Step 5: Fix Casing Inconsistencies

When fixing casing:

1. For variables and parameters: use `snake_case`
2. For classes: use `PascalCase`
3. For constants: use `UPPER_SNAKE_CASE`

Example approach:

```python
# Before
def processDocument(documentContent, outputPath):
    # Implementation...

# After
def process_document(document_content, output_path):
    # Implementation...
```

## Step 6: Batch Refactoring

For large-scale renaming, consider using automated refactoring tools:

1. Use an IDE's refactoring tools (e.g., PyCharm's "Rename")
2. Use automated scripts for batch renaming
3. Consider using the `rope` library for Python refactoring

Example script for batch renaming parameters:

```python
import ast
import astor
import re
import os

def rename_parameters(file_path, old_param, new_param):
    with open(file_path, 'r') as file:
        source = file.read()
    
    tree = ast.parse(source)
    
    class ParameterRenamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            for arg in node.args.args:
                if arg.arg == old_param:
                    arg.arg = new_param
            
            self.generic_visit(node)
            return node
    
    new_tree = ParameterRenamer().visit(tree)
    new_source = astor.to_source(new_tree)
    
    with open(file_path, 'w') as file:
        file.write(new_source)
```

## Step 7: Testing

After each batch of renaming:

1. Run the unit tests to ensure nothing is broken
2. Run the integration tests to verify everything works together
3. Verify that imports are still working correctly
4. Run the application to ensure no runtime errors

## Example Refactoring Plan

Here's a concrete plan for refactoring naming conventions:

1. **Phase 1: Fix ambiguous method names in public interfaces**
   - Replace `process` with more specific names like `process_document`
   - Replace `get` with more specific names like `get_completion_for_analysis`
   - Replace `handle` with more specific names like `handle_validation_error`

2. **Phase 2: Standardize parameter names**
   - Replace `file` with `file_path`
   - Replace `output` with `output_path`
   - Replace `model` with `model_name`

3. **Phase 3: Fix casing inconsistencies**
   - Convert camelCase method names to snake_case
   - Convert snake_case class names to PascalCase
   - Convert mixed-case variables to snake_case

4. **Phase 4: Fix file naming**
   - Rename files to follow snake_case convention
   - Update all imports accordingly

## Tracking Progress

Create a tracking document to record progress:

```markdown
# Naming Refactoring Progress

## Completed
- [x] Renamed ambiguous methods in LLMClient
- [x] Standardized parameter names in DocumentProcessor

## In Progress
- [ ] Standardizing parameter names in ReferenceValidator
- [ ] Converting camelCase methods in ScopusClient to snake_case

## To Do
- [ ] Fix file naming inconsistencies
- [ ] Update documentation references
```

## Handling Breaking Changes

If renaming would cause breaking changes to external code:

1. Add deprecation warnings for the old names
2. Create wrapper methods that call the new methods
3. Document the changes in the API documentation
4. Plan for removal of deprecated methods in a future version

Example:

```python
import warnings

def get_data(file):
    """
    DEPRECATED: Use get_document_data(file_path) instead.
    This method will be removed in version 2.0.0.
    """
    warnings.warn(
        "get_data is deprecated, use get_document_data instead",
        DeprecationWarning,
        stacklevel=2
    )
    return get_document_data(file)

def get_document_data(file_path):
    """Get data from a document file."""
    # Implementation...
```