# Tests for Paper Revision Tool

This directory contains tests for the Paper Revision Tool. The tests are organized by type and component, following the testing pyramid approach.

## Test Structure

```
tests/
├── unit/               # Unit tests for individual components
│   ├── domain/         # Tests for domain entities
│   ├── services/       # Tests for service layer
│   ├── adapters/       # Tests for adapter layer
│   ├── utils/          # Tests for utility functions
│   └── config/         # Tests for configuration system
├── integration/        # Integration tests between components
│   ├── service_flow/   # Tests for service layer integration
│   ├── adapter_flow/   # Tests for adapter layer integration
│   └── api_flow/       # Tests for API integration
├── functional/         # Functional tests for user workflows
├── fixtures/           # Test fixtures and test data
│   ├── data/           # Test data files
│   ├── mocks/          # Mock objects and responses
│   └── factories.py    # Factory functions for test objects
├── conftest.py         # Pytest configuration and shared fixtures
└── README.md           # This file
```

## Running Tests

### Running all tests

```bash
pytest
```

### Running specific test types

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run functional tests only
pytest tests/functional/
```

### Running tests for specific components

```bash
# Run tests for domain entities
pytest tests/unit/domain/

# Run tests for service layer
pytest tests/unit/services/
```

### Running tests with coverage

```bash
# Run tests with coverage report
pytest --cov=src

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

## Test Naming Conventions

- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`
- Test fixtures should be named with descriptive names of what they provide

## Writing Tests

### Unit Tests

Unit tests should:
- Test a single unit of functionality
- Have clear assertions
- Be independent of other tests
- Use mocks or stubs for external dependencies
- Be fast to execute

Example:

```python
def test_paper_creation():
    # Arrange
    title = "Test Paper"
    authors = ["Author 1", "Author 2"]
    
    # Act
    paper = Paper(title=title, authors=authors)
    
    # Assert
    assert paper.title == title
    assert paper.authors == authors
```

### Integration Tests

Integration tests should:
- Test interactions between components
- Verify that components work together
- Use real implementations when possible
- Mock external services when necessary

### Functional Tests

Functional tests should:
- Test complete user workflows
- Verify that the system works as a whole
- Test from the user's perspective
- Use as few mocks as possible

## Test Fixtures

Test fixtures provide test data and objects for tests. They are defined in:
- `tests/fixtures/` directory for specialized fixtures
- `conftest.py` for shared fixtures

## Mocking

Tests should use mocks for:
- External API calls
- File system operations
- Database operations
- Any other external dependencies

Example using the `unittest.mock` library:

```python
from unittest.mock import patch, Mock

def test_llm_client_gets_completion():
    # Arrange
    mock_response = "Mocked completion response"
    
    # Mock the API call
    with patch('src.utils.llm_client.anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.completions.create.return_value.completion = mock_response
        
        # Act
        client = AnthropicClient(api_key="fake_key")
        result = client.get_completion("Test prompt")
        
        # Assert
        assert result == mock_response
        mock_client.completions.create.assert_called_once()
```

## Continuous Integration

Tests are run automatically on each commit through the CI/CD pipeline. Tests must pass before code can be merged.