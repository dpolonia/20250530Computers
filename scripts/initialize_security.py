#!/usr/bin/env python3
"""
Security initialization script for Paper Revision System.

This script sets up the security infrastructure for the system, including:
- Creating secure directories for credentials
- Setting up the encryption key for credential storage
- Initializing security settings
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.credential_manager import get_credential_manager
from src.security.input_validator import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("security_init")


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Initialize security settings for Paper Revision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic initialization
  python initialize_security.py
  
  # Force reinitialization
  python initialize_security.py --force
  
  # Custom configuration directory
  python initialize_security.py --config-dir /path/to/secure/config
"""
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reinitialization of security settings"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Custom configuration directory for security settings"
    )
    
    return parser


def create_secure_directories(config_dir=None):
    """Create secure directories for credentials and configuration.
    
    Args:
        config_dir: Optional custom configuration directory
        
    Returns:
        Dictionary with created directories
    """
    dirs = {}
    
    # Determine base directory
    base_dir = Path.home() / ".paper_revision_secure"
    if config_dir:
        base_dir = Path(config_dir)
    
    dirs["base"] = base_dir
    
    # Ensure the base directory exists with secure permissions
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
        logger.info(f"Created base security directory: {base_dir}")
    
    # Set secure permissions (rwx------ / 700)
    try:
        os.chmod(base_dir, 0o700)
        logger.info(f"Set secure permissions on {base_dir}")
    except Exception as e:
        logger.warning(f"Could not set secure permissions on {base_dir}: {e}")
    
    # Create subdirectories
    subdirs = ["credentials", "logs", "keys", "temp"]
    for subdir in subdirs:
        dir_path = base_dir / subdir
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Set secure permissions
        try:
            os.chmod(dir_path, 0o700)
        except Exception as e:
            logger.warning(f"Could not set secure permissions on {dir_path}: {e}")
        
        dirs[subdir] = dir_path
    
    return dirs


def initialize_credential_manager(config_dir=None):
    """Initialize the credential manager.
    
    Args:
        config_dir: Optional custom configuration directory
        
    Returns:
        The initialized credential manager
    """
    try:
        # Get credential manager (this will initialize it)
        if config_dir:
            from src.security.credential_manager import CredentialManager
            credential_manager = CredentialManager(config_dir)
        else:
            credential_manager = get_credential_manager()
        
        logger.info("Credential manager initialized successfully")
        return credential_manager
    except Exception as e:
        logger.error(f"Error initializing credential manager: {e}")
        raise


def create_default_security_config(base_dir):
    """Create a default security configuration file.
    
    Args:
        base_dir: Base security directory
        
    Returns:
        Path to the created configuration file
    """
    config_file = base_dir / "security_config.json"
    
    # Only create if it doesn't exist
    if not config_file.exists():
        import json
        
        # Default security configuration
        default_config = {
            "logging": {
                "level": "INFO",
                "mask_patterns": ["api_key", "password", "token", "secret"],
                "log_file": str(base_dir / "logs" / "security.log"),
                "max_size_mb": 10,
                "backup_count": 5
            },
            "file_security": {
                "temp_directory": str(base_dir / "temp"),
                "temp_file_ttl": 3600,  # seconds
                "allowed_extensions": [".pdf", ".docx", ".txt", ".bib", ".json"],
                "max_upload_size_mb": 20
            },
            "api_security": {
                "rate_limit_enabled": True,
                "rate_limit_requests": 100,
                "rate_limit_period": 3600,  # seconds
                "require_api_key": True
            },
            "database": {
                "encryption_enabled": False,
                "backup_enabled": True,
                "backup_interval_hours": 24,
                "max_backups": 10
            },
            "last_updated": None
        }
        
        # Write the configuration file
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)
        
        # Set secure permissions
        os.chmod(config_file, 0o600)
        
        logger.info(f"Created default security configuration: {config_file}")
    else:
        logger.info(f"Security configuration already exists: {config_file}")
    
    return config_file


def verify_environment():
    """Verify the system environment for security requirements."""
    # Check Python version
    import platform
    python_version = platform.python_version_tuple()
    if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 7):
        logger.warning("Recommended Python version is 3.7 or higher for security features")
    
    # Check for required security packages
    required_packages = ["cryptography", "bcrypt"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing security packages: {', '.join(missing_packages)}")
        logger.warning("Install them with: pip install " + " ".join(missing_packages))
    
    # Check temporary directory permissions
    import tempfile
    temp_dir = tempfile.gettempdir()
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        os.unlink(temp_file.name)
        logger.info(f"Temporary directory ({temp_dir}) is writable")
    except Exception as e:
        logger.warning(f"Issue with temporary directory ({temp_dir}): {e}")


def main():
    """Main function for security initialization."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        logger.info("Starting security initialization")
        
        # Verify environment
        verify_environment()
        
        # Create secure directories
        dirs = create_secure_directories(args.config_dir)
        
        # Initialize credential manager
        credential_manager = initialize_credential_manager(
            str(dirs["credentials"]) if "credentials" in dirs else None
        )
        
        # Create default security configuration
        config_file = create_default_security_config(dirs["base"])
        
        logger.info("Security initialization completed successfully")
        logger.info(f"Security base directory: {dirs['base']}")
        logger.info(f"Credential storage: {dirs['credentials']}")
        logger.info(f"Security configuration: {config_file}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Store your API keys using scripts/manage_credentials.py")
        logger.info("2. Review and customize the security configuration")
        logger.info("3. Run a security verification with scripts/verify_security_config.py")
        
        return 0
    
    except ValidationError as e:
        logger.error(f"Validation error during security initialization: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during security initialization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())