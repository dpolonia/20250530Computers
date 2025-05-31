# Testing Guide

This guide provides instructions for maintaining and extending the test suite for the Paper Revision System. It covers best practices, patterns, and guidelines for keeping the test suite comprehensive and effective.

## Test Coverage Goals

- **Domain Entities**: 100% coverage of all domain entity functionality
- **Services**: 90%+ coverage of service methods with focus on business logic paths
- **Adapters**: 80%+ coverage with focus on data transformation logic
- **Workflows**: End-to-end flows covered by functional tests

## Adding New Tests

### 1. Determine Test Type

Choose the appropriate test type based on what you're testing:

- **Unit Tests**: For testing isolated components (classes, functions)
- **Integration Tests**: For testing interactions between components
- **Functional Tests**: For testing complete workflows

### 2. Follow the Test Directory Structure

Place your test in the appropriate directory:

```
tests/
├── unit/               # Unit tests (test single components in isolation)
│   ├── domain/         # Domain entity tests
│   ├── services/       # Service class tests
│   ├── adapters/       # Adapter tests
│   ├── utils/          # Utility function tests
│   └── ...
├── integration/        # Integration tests (test component interactions)
│   ├── service_flow/   # Service-to-service interaction tests
│   ├── adapter_flow/   # Adapter-to-service interaction tests
│   └── api_flow/       # API gateway interaction tests
└── functional/         # Functional tests (test complete workflows)
```

### 3. Use Fixtures and Factories

Leverage existing fixtures and factories to create test data:

```python
def test_new_feature(sample_paper, paper_service):
    # Use the fixtures directly in your test
    result = paper_service.new_feature(sample_paper)
    assert result is not None
```

Create new fixtures in `conftest.py` for shared objects or in test files for specific cases.

### 4. Follow the AAA Pattern

Structure tests using the Arrange-Act-Assert pattern:

```python
def test_something():
    # Arrange - setup test data and conditions
    paper = create_paper(title="Test Title")
    
    # Act - execute the code being tested
    result = paper.get_section_count()
    
    # Assert - verify the results
    assert result == len(paper.sections)
```

### 5. Mocking External Dependencies

Use the `unittest.mock` library to mock external dependencies:

```python
@patch('src.utils.llm_client.Anthropic')
def test_with_mock(mock_anthropic_class):
    mock_anthropic_instance = Mock()
    mock_anthropic_class.return_value = mock_anthropic_instance
    mock_anthropic_instance.completions.create.return_value = {"completion": "test response"}
    
    client = AnthropicClient(api_key="fake_key")
    result = client.complete("test prompt")
    
    assert result == "test response"
```

## Test Naming Conventions

- Test file names: `test_[module_name].py`
- Test function names: `test_[method_name]_[scenario]`
- Test class names: `Test[ClassBeingTested]`

Examples:
- `test_paper.py`
- `test_extract_title_with_valid_input`
- `TestPaperService`

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Specific Test Categories

```bash
# Run all unit tests
pytest tests/unit/

# Run service tests
pytest tests/unit/services/

# Run a specific test file
pytest tests/unit/domain/test_paper.py

# Run a specific test function
pytest tests/unit/domain/test_paper.py::test_paper_initialization
```

### Test Coverage

Check test coverage with:

```bash
pytest --cov=src
```

Generate an HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on the state from other tests
2. **Descriptive Test Names**: Name tests clearly to describe what they're testing
3. **Focused Tests**: Each test should verify one specific aspect of functionality
4. **Avoid Test Logic**: Minimize conditional logic in tests
5. **Maintainable Test Data**: Use fixtures and factories to create test data
6. **Test Edge Cases**: Include tests for error conditions and edge cases
7. **Keep Tests Fast**: Optimize for speed to encourage frequent testing

## When to Add Tests

Add or update tests when:

- Creating new features
- Fixing bugs (add a test that would have caught the bug)
- Refactoring code (ensure existing functionality is preserved)
- Extending existing functionality

## Continuous Integration

Tests are automatically run in the CI pipeline on each commit. Make sure your tests pass locally before pushing changes:

```bash
# Run quick check before committing
pytest
```

## Troubleshooting Common Test Issues

### Test Dependencies

If your test depends on specific fixtures or mock objects:

```python
@pytest.mark.usefixtures("required_fixture")
def test_with_dependencies():
    # This test requires required_fixture to be set up
    ...
```

### Handling Asynchronous Code

For testing async code:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Parameterized Tests

For testing multiple input variations:

```python
@pytest.mark.parametrize("input,expected", [
    ("input1", "expected1"),
    ("input2", "expected2"),
])
def test_parameterized(input, expected):
    result = function_under_test(input)
    assert result == expected
```

## Extending the Test Suite

When extending the system with new components:

1. Create corresponding test files in the appropriate directories
2. Add fixtures for new domain entities in `conftest.py` or `fixtures/factories.py`
3. Create mocks for new external dependencies
4. Add integration tests for new component interactions
5. Update functional tests to cover new workflows

Remember: Tests are a form of documentation. Clear, well-structured tests help future developers understand how the system works.