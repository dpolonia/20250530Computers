# Naming Conventions Style Guide

This document defines the standard naming conventions for the Paper Revision Tool project. Consistent naming makes the codebase more readable and maintainable.

## General Principles

1. Be descriptive and specific
2. Avoid abbreviations unless they are widely understood
3. Be consistent within and across modules
4. Prioritize clarity over brevity

## Language-Specific Conventions

### Python

#### Files and Modules

- Use `snake_case` for file and module names
- Make names descriptive of the module's content
- Avoid generic names like `utils.py` (prefer `path_utils.py`, `text_utils.py`, etc.)

```python
# Good
document_processor.py
reference_validator.py

# Bad
utils.py
processor.py
```

#### Classes

- Use `PascalCase` (also known as `UpperCamelCase`) for class names
- Names should be nouns or noun phrases
- Be specific about the class's purpose

```python
# Good
class DocumentProcessor:
class PaperAnalyzer:
class ReferenceValidator:

# Bad
class Processor:
class Manager:
class Helper:
```

#### Functions and Methods

- Use `snake_case` for function and method names
- Names should be verbs or verb phrases
- Be specific about what the function does
- Avoid generic terms like "process", "handle", "get" without context

```python
# Good
def extract_references_from_text(text):
def validate_doi_format(doi):
def analyze_paper_structure(paper):

# Bad
def process(data):
def handle_text(text):
def get_result():
```

#### Variables and Parameters

- Use `snake_case` for variable and parameter names
- Be descriptive about what the variable contains
- Use consistent names for the same concept across different functions

```python
# Good
paper_text = extract_text(paper_path)
reference_count = len(references)
doi_pattern = re.compile(r'10\.\d{4,}\/\S+')

# Bad
txt = extract_text(p)
n = len(refs)
pattern = re.compile(r'10\.\d{4,}\/\S+')
```

#### Constants

- Use `UPPER_SNAKE_CASE` for constants
- Place constants at the module level or in a dedicated constants module

```python
# Good
MAX_REFERENCES = 100
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# Bad
maxReferences = 100
default_timeout = 30
apiBaseUrl = "https://api.example.com"
```

#### Private Members

- Prefix private methods and attributes with a single underscore
- Use double underscores only for name mangling when necessary

```python
# Good
class ReferenceValidator:
    def _validate_format(self, reference):
        # Internal implementation
        
    def validate(self, reference):
        # Public method
        
# Bad
class ReferenceValidator:
    def validateFormat(self, reference):
        # Inconsistent case style
        
    def __validate(self, reference):
        # Unnecessary name mangling
```

## Domain-Specific Naming Conventions

### API Operations

- Use verb-noun format for API operations
- Be consistent with verb choice

```python
# Good
get_completion()
extract_sections()
validate_references()

# Bad
completion_getter()
section_extraction()
check_refs()
```

### LLM Operations

- Use the `get_completion` pattern for LLM completion methods
- Include context in method names

```python
# Good
get_completion_for_paper_analysis()
get_completion_for_reference_validation()

# Bad
get_llm_result()
completion()
```

### File Operations

- Use standard verbs for file operations: `read`, `write`, `create`, `update`, `delete`
- Include the file type in the method name

```python
# Good
read_pdf()
write_docx()
create_changes_document()

# Bad
process_file()
handle_document()
output()
```

## Parameter Naming Conventions

### Common Parameters

Use consistent names for common parameters across the codebase:

| Concept | Standard Name | Type |
|---------|--------------|------|
| File path | `file_path` | `str` |
| Directory path | `directory_path` | `str` |
| Text content | `text` or `content` | `str` |
| API key | `api_key` | `str` |
| Model name | `model_name` | `str` |
| Output path | `output_path` | `str` |
| Paper | `paper` | `Paper` |
| References | `references` | `List[Reference]` |
| Changes | `changes` | `List[Change]` |
| Context | `context` | `RevisionContext` |

### Function Parameters

- Order parameters from most important to least important
- Place required parameters before optional parameters
- Group related parameters together
- Use consistent parameter order across similar functions

```python
# Good - Consistent parameter order
def create_document(content, output_path, format_type="docx", overwrite=False):
def create_report(content, output_path, format_type="pdf", overwrite=False):

# Bad - Inconsistent parameter order
def create_document(content, format_type="docx", output_path, overwrite=False):
def create_report(output_path, content, overwrite=False, format_type="pdf"):
```

## Naming Anti-Patterns to Avoid

### 1. Ambiguous Names

Avoid generic names that don't convey specific meaning:

```python
# Avoid
def process(data):
def handle(item):
def get_data():

# Prefer
def extract_citations_from_text(text):
def handle_reference_validation_error(error):
def get_paper_metadata():
```

### 2. Inconsistent Casing

Don't mix snake_case and camelCase:

```python
# Avoid
def process_document(docContent, outputPath):

# Prefer
def process_document(doc_content, output_path):
```

### 3. Misleading Names

Avoid names that suggest different functionality than what is provided:

```python
# Avoid
def delete_file(file_path):  # If it actually just archives the file

# Prefer
def archive_file(file_path):
```

### 4. Abbreviations

Avoid unclear abbreviations:

```python
# Avoid
def proc_doc(doc):
def val_ref(ref):

# Prefer
def process_document(document):
def validate_reference(reference):
```

## Implementing Naming Conventions

When applying these naming conventions:

1. Start with new code
2. Update existing code when modifying those files
3. Focus on public interfaces first
4. Document any exceptions to these rules

For large-scale renaming, use the `rename_checker.py` script to identify and correct inconsistencies.