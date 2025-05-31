# Security Configuration Guide

This guide explains how to configure security settings for the Paper Revision System. Follow these steps to ensure your deployment is secure.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [API Key Configuration](#api-key-configuration)
3. [Database Security Settings](#database-security-settings)
4. [File System Security](#file-system-security)
5. [Logging Configuration](#logging-configuration)
6. [Environment Variables](#environment-variables)
7. [Authentication Configuration](#authentication-configuration)
8. [Firewall Settings](#firewall-settings)

## Initial Setup

Before running the system in production, perform these initial security setup steps:

1. Create a secure environment:

```bash
# Create a dedicated user for running the application
sudo useradd -m -s /bin/bash paperrevsys

# Set proper permissions
sudo chown -R paperrevsys:paperrevsys /path/to/application

# Restrict access to configuration files
sudo chmod 750 /path/to/application/config
```

2. Install security dependencies:

```bash
pip install -r requirements.txt
pip install -r security_requirements.txt
```

3. Run the security initialization script:

```bash
python scripts/initialize_security.py
```

## API Key Configuration

### Securely Storing API Keys

Use the credential management tool to store API keys:

```bash
# Store Anthropic API key
python scripts/manage_credentials.py store anthropic

# Store OpenAI API key
python scripts/manage_credentials.py store openai

# Store Google API key
python scripts/manage_credentials.py store google
```

### Verify API Key Configuration

Check that API keys are stored correctly:

```bash
python scripts/manage_credentials.py verify
```

### API Key Rotation

Set up a reminder to rotate API keys regularly:

```bash
# Add to your crontab (sends email reminder every 90 days)
(crontab -l ; echo "0 9 1 */3 * /path/to/application/scripts/remind_key_rotation.sh") | crontab -
```

## Database Security Settings

### SQLite Configuration

SQLite database security is configured in the `WorkflowDB` class. The following pragmas are set:

- `PRAGMA journal_mode=WAL`: Write-Ahead Logging for crash recovery
- `PRAGMA foreign_keys=ON`: Enforce foreign key constraints
- `PRAGMA secure_delete=ON`: Securely delete data

### File Permissions

Ensure the database file has proper permissions:

```bash
# Set database file permissions to be readable only by the application user
sudo chmod 600 /path/to/application/.cache/workflow.db
sudo chmod 700 /path/to/application/.cache
```

### Backup Security

Secure your database backups:

```bash
# Create an encrypted backup
python scripts/backup_db.py --encrypt
```

## File System Security

### Directory Structure

Create a secure directory structure with appropriate permissions:

```bash
# Application directories
sudo mkdir -p /path/to/application/data
sudo mkdir -p /path/to/application/logs
sudo mkdir -p /path/to/application/temp

# Set permissions
sudo chown -R paperrevsys:paperrevsys /path/to/application
sudo chmod 750 /path/to/application
sudo chmod 700 /path/to/application/data
sudo chmod 700 /path/to/application/logs
sudo chmod 700 /path/to/application/temp
```

### Temporary Files

Configure secure temporary file handling:

```bash
# Edit config.json
{
  "temp_directory": "/path/to/application/temp",
  "temp_file_ttl": 3600,  // Time to live in seconds
  "clean_temp_on_startup": true
}
```

## Logging Configuration

### Secure Logging

Configure secure logging that masks sensitive information:

```python
# Edit logging_config.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'filters': {
        'mask_sensitive_data': {
            '()': 'src.security.logging_filters.MaskSensitiveDataFilter',
            'patterns': ['api_key', 'password', 'token', 'secret']
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/path/to/application/logs/app.log',
            'formatter': 'standard',
            'filters': ['mask_sensitive_data']
        },
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    }
}
```

### Log Rotation

Set up log rotation to prevent log files from growing too large:

```bash
# Add to /etc/logrotate.d/paperrevsys
/path/to/application/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 paperrevsys paperrevsys
    sharedscripts
    postrotate
        systemctl reload paperrevsys.service
    endscript
}
```

## Environment Variables

### Production Environment

Set up environment variables for production:

```bash
# Create a .env.production file (do not commit to version control)
export FLASK_ENV=production
export DEBUG=False
export LOG_LEVEL=INFO
export TEMP_DIR=/path/to/application/temp
export DATA_DIR=/path/to/application/data
export SECURITY_LEVEL=high
```

### Development Environment

Set up environment variables for development:

```bash
# Create a .env.development file (do not commit to version control)
export FLASK_ENV=development
export DEBUG=True
export LOG_LEVEL=DEBUG
export TEMP_DIR=/path/to/application/temp
export DATA_DIR=/path/to/application/data
export SECURITY_LEVEL=medium
```

## Authentication Configuration

### API Authentication

Configure API authentication:

```python
# Edit config/auth.py
AUTH_CONFIG = {
    'api_key_header': 'X-API-Key',
    'api_key_query_param': 'api_key',
    'api_key_expiry_days': 90,
    'rate_limit_enabled': True,
    'rate_limit_requests': 100,
    'rate_limit_period': 3600,  # seconds
    'ip_whitelist': ['127.0.0.1', '192.168.1.0/24']
}
```

### User Authentication

Configure user authentication if used:

```python
# Edit config/auth.py
USER_AUTH_CONFIG = {
    'session_expiry': 3600,  # seconds
    'max_failed_attempts': 5,
    'lockout_period': 300,  # seconds
    'password_min_length': 12,
    'password_require_mixed_case': True,
    'password_require_numbers': True,
    'password_require_special_chars': True
}
```

## Firewall Settings

### Application Firewall

Configure the application firewall rules:

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # Flask API port
sudo ufw enable
```

### Database Firewall

Configure database connection restrictions:

```bash
# For external databases (if used instead of SQLite)
sudo ufw allow from 192.168.1.0/24 to any port 5432  # PostgreSQL
```

### API Rate Limiting

Configure API rate limiting:

```python
# Edit config/api.py
RATE_LIMIT_CONFIG = {
    'enabled': True,
    'default_limits': ['100 per hour', '2000 per day'],
    'storage_uri': 'memory://',
    'key_prefix': 'paper_revision_limiter'
}
```

---

After completing all configuration steps, run the security verification script:

```bash
python scripts/verify_security_config.py
```

This script will check your configuration and report any potential security issues.