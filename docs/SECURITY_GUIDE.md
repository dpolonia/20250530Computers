# Security Guide for Paper Revision System

This guide outlines the security measures implemented in the Paper Revision System, along with best practices for maintaining and extending the system securely.

## Table of Contents

1. [Introduction](#introduction)
2. [Security Architecture](#security-architecture)
3. [API Key Management](#api-key-management)
4. [Database Security](#database-security)
5. [Input Validation](#input-validation)
6. [Error Handling](#error-handling)
7. [File Security](#file-security)
8. [Secure Development Practices](#secure-development-practices)
9. [Vulnerability Reporting](#vulnerability-reporting)

## Introduction

The Paper Revision System processes sensitive academic content and uses third-party APIs that require secure credential management. This guide documents the security measures implemented to protect this information and provides guidance for developers extending the system.

## Security Architecture

The system's security architecture is built on several key components:

- **Credential Manager**: Secure storage and retrieval of API keys
- **Input Validation**: Comprehensive validation of all user inputs
- **Query Sanitization**: Prevention of SQL injection vulnerabilities
- **Secure File Handling**: Validation and safe handling of files
- **Error Handling**: Proper handling of errors without information leakage
- **Logging**: Secure logging that masks sensitive information

The security modules are located in the `src/security` directory:

- `credential_manager.py`: Secure API key storage and retrieval
- `input_validator.py`: Input validation utilities
- `query_sanitizer.py`: Database query sanitization

## API Key Management

The system uses a secure credential manager for API keys:

### Encryption and Storage

- API keys are encrypted using Fernet symmetric encryption (based on AES-128-CBC)
- Keys are stored in a secure location with restricted permissions (700/600)
- Each machine generates a unique encryption key derived from hardware identifiers

### Managing API Keys

Use the `manage_credentials.py` script for key management:

```bash
# Store a new API key
python scripts/manage_credentials.py store anthropic

# Verify API keys
python scripts/manage_credentials.py verify

# Export keys to environment variables (temporary)
python scripts/manage_credentials.py export

# Import keys from environment variables
python scripts/manage_credentials.py import

# List stored keys (masked)
python scripts/manage_credentials.py list

# Delete a key
python scripts/manage_credentials.py delete openai
```

### Best Practices

1. **Never hardcode API keys** in source code
2. **Don't log API keys** or include them in error messages
3. **Rotate API keys** periodically (every 90 days)
4. **Use environment variables** only for local development
5. **Validate API keys** before using them
6. **Limit API key permissions** to what's necessary

## Database Security

The system implements several measures to prevent SQL injection and ensure database security:

### Parameterized Queries

All database operations use parameterized queries to prevent SQL injection attacks:

```python
# Example of secure query creation
query, params = create_parameterized_query(
    table_name="users",
    columns=["id", "username", "email"],
    where_conditions={"active": True},
    order_by=["username ASC"],
    limit=10
)

# Example of secure query execution
results = execute_query(connection, query, params)
```

### Table and Column Name Sanitization

Table and column names are sanitized to prevent injection attacks:

```python
safe_table = sanitize_table_name(user_supplied_table_name)
safe_column = sanitize_column_name(user_supplied_column_name)
```

### Database Configuration

The SQLite database is configured with security-enhancing options:

- `PRAGMA journal_mode=WAL`: Write-Ahead Logging for crash recovery
- `PRAGMA foreign_keys=ON`: Enforce foreign key constraints
- `PRAGMA secure_delete=ON`: Securely delete data

### Best Practices

1. **Always use parameterized queries** for all database operations
2. **Sanitize table and column names** when they come from user input
3. **Use the least privilege principle** for database accounts
4. **Validate all input** before using it in database operations
5. **Handle database errors** without exposing implementation details

## Input Validation

The system implements comprehensive input validation:

### Validation Functions

The `input_validator.py` module provides specialized validation functions:

- `validate_string_input()`: For text inputs
- `validate_numeric_input()`: For numeric inputs
- `validate_file_path()`: For file paths
- `validate_directory_path()`: For directory paths
- `validate_api_key()`: For API keys
- `validate_doi()`: For Digital Object Identifiers
- `validate_email()`: For email addresses
- `validate_dict_schema()`: For dictionary validation against a schema

### Validation Decorator

Use the validation decorator to simplify function parameter validation:

```python
@validate_input(
    path=lambda x: validate_file_path(x, 'pdf'),
    value=lambda x: validate_numeric_input(x, min_value=0)
)
def process_file(path, value):
    # Implementation here
```

### Best Practices

1. **Validate all user inputs** before processing
2. **Use appropriate validation functions** for different types of data
3. **Validate at system boundaries** (API endpoints, CLI interfaces)
4. **Check both format and content validity**
5. **Apply context-specific validation** for special data types

## Error Handling

The system implements secure error handling to prevent information leakage:

### Custom Exception Classes

- `ValidationError`: For input validation errors
- `QuerySanitizationError`: For database query errors

### Best Practices

1. **Use custom exception classes** for different error types
2. **Log detailed errors** for troubleshooting
3. **Return generic error messages** to users
4. **Don't expose implementation details** in error messages
5. **Log full stack traces** only in development or in secure logs

## File Security

The system implements secure file handling:

### File Path Validation

File paths are validated before use:

```python
safe_path = validate_file_path(
    user_supplied_path,
    file_type='pdf',
    must_exist=True,
    readable=True
)
```

### Directory Path Validation

Directory paths are validated before use:

```python
safe_dir = validate_directory_path(
    user_supplied_dir,
    must_exist=False,
    create_if_missing=True,
    writable=True
)
```

### Best Practices

1. **Validate all file paths** before use
2. **Use absolute paths** to avoid directory traversal attacks
3. **Check file extensions** for expected file types
4. **Restrict file operations** to specific directories
5. **Apply proper permissions** to sensitive files

## Secure Development Practices

Follow these practices when extending the system:

### Code Quality

1. **Use static analysis tools**: pylint, mypy, bandit
2. **Write unit tests** for security-critical code
3. **Conduct code reviews** with security focus
4. **Keep dependencies updated**

### Authentication and Authorization

1. **Implement proper authentication** for APIs and UIs
2. **Use the principle of least privilege**
3. **Validate permissions** before operations

### Sensitive Data

1. **Minimize sensitive data collection**
2. **Don't store sensitive data unless necessary**
3. **Mask sensitive data in logs**

## Vulnerability Reporting

If you discover a security vulnerability in the Paper Revision System:

1. **Do not disclose publicly** until addressed
2. **Report to the security team** at security@example.com
3. **Include detailed information** about the vulnerability
4. **Provide steps to reproduce** if possible

## Security Updates

The security team will:

1. **Acknowledge receipt** within 48 hours
2. **Provide a timeline** for addressing the vulnerability
3. **Issue security updates** as needed
4. **Notify users** of security-critical updates