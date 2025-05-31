#!/usr/bin/env python3
"""
Security configuration verification script for Paper Revision System.

This script verifies the security configuration of the system, including:
- Credential manager settings
- File permissions
- Database security settings
- Input validation
- Error handling
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.credential_manager import get_credential_manager, CredentialManager
from src.security.input_validator import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("security_verify")


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Verify security configuration for Paper Revision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python verify_security_config.py
  
  # Verify with detailed output
  python verify_security_config.py --verbose
  
  # Custom configuration directory
  python verify_security_config.py --config-dir /path/to/secure/config
"""
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed verification output"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Custom configuration directory for security settings"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix security issues automatically"
    )
    
    return parser


def verify_credential_manager(config_dir=None, verbose=False, fix=False):
    """Verify the credential manager configuration.
    
    Args:
        config_dir: Optional custom configuration directory
        verbose: Whether to show detailed output
        fix: Whether to attempt to fix issues automatically
        
    Returns:
        Tuple of (success, issues)
    """
    issues = []
    success = True
    
    try:
        # Get credential manager
        if config_dir:
            from src.security.credential_manager import CredentialManager
            credential_manager = CredentialManager(config_dir)
        else:
            credential_manager = get_credential_manager()
        
        # Verify credential directory
        cred_dir = Path(credential_manager.config_dir)
        if verbose:
            logger.info(f"Credential directory: {cred_dir}")
        
        # Check if directory exists
        if not cred_dir.exists():
            issues.append(f"Credential directory does not exist: {cred_dir}")
            success = False
            
            # Try to fix
            if fix:
                try:
                    cred_dir.mkdir(parents=True)
                    os.chmod(cred_dir, 0o700)
                    logger.info(f"Created credential directory: {cred_dir}")
                except Exception as e:
                    logger.error(f"Failed to create credential directory: {e}")
        
        # Check directory permissions
        if cred_dir.exists():
            try:
                dir_stats = os.stat(cred_dir)
                dir_mode = dir_stats.st_mode & 0o777
                
                if dir_mode != 0o700:
                    issues.append(f"Credential directory has incorrect permissions: {oct(dir_mode)}, should be 0o700")
                    success = False
                    
                    # Try to fix
                    if fix:
                        try:
                            os.chmod(cred_dir, 0o700)
                            logger.info(f"Fixed permissions on credential directory: {cred_dir}")
                        except Exception as e:
                            logger.error(f"Failed to fix permissions on credential directory: {e}")
            except Exception as e:
                issues.append(f"Could not check credential directory permissions: {e}")
                success = False
        
        # Check encryption key
        key_file = cred_dir / "encryption.key"
        if not key_file.exists():
            issues.append("Encryption key does not exist")
            success = False
            
            # Try to fix
            if fix:
                try:
                    # This will generate a new key
                    key = credential_manager._get_or_create_encryption_key()
                    logger.info("Generated new encryption key")
                except Exception as e:
                    logger.error(f"Failed to generate encryption key: {e}")
        
        # Check key file permissions
        if key_file.exists():
            try:
                key_stats = os.stat(key_file)
                key_mode = key_stats.st_mode & 0o777
                
                if key_mode != 0o600:
                    issues.append(f"Encryption key file has incorrect permissions: {oct(key_mode)}, should be 0o600")
                    success = False
                    
                    # Try to fix
                    if fix:
                        try:
                            os.chmod(key_file, 0o600)
                            logger.info(f"Fixed permissions on encryption key file: {key_file}")
                        except Exception as e:
                            logger.error(f"Failed to fix permissions on encryption key file: {e}")
            except Exception as e:
                issues.append(f"Could not check encryption key file permissions: {e}")
                success = False
    
    except Exception as e:
        issues.append(f"Error verifying credential manager: {e}")
        success = False
    
    if verbose:
        if success:
            logger.info("Credential manager verification passed")
        else:
            logger.warning("Credential manager verification failed")
            for issue in issues:
                logger.warning(f"- {issue}")
    
    return success, issues


def verify_security_config(config_dir=None, verbose=False, fix=False):
    """Verify the security configuration.
    
    Args:
        config_dir: Optional custom configuration directory
        verbose: Whether to show detailed output
        fix: Whether to attempt to fix issues automatically
        
    Returns:
        Tuple of (success, issues)
    """
    issues = []
    success = True
    
    try:
        # Determine base directory
        base_dir = Path.home() / ".paper_revision_secure"
        if config_dir:
            base_dir = Path(config_dir)
        
        # Check if directory exists
        if not base_dir.exists():
            issues.append(f"Security configuration directory does not exist: {base_dir}")
            success = False
            
            # Try to fix
            if fix:
                try:
                    from scripts.initialize_security import create_secure_directories
                    dirs = create_secure_directories(config_dir)
                    logger.info(f"Created security directories: {dirs}")
                except Exception as e:
                    logger.error(f"Failed to create security directories: {e}")
                    
            return success, issues
        
        # Check for security configuration file
        config_file = base_dir / "security_config.json"
        if not config_file.exists():
            issues.append(f"Security configuration file does not exist: {config_file}")
            success = False
            
            # Try to fix
            if fix:
                try:
                    from scripts.initialize_security import create_default_security_config
                    config_file = create_default_security_config(base_dir)
                    logger.info(f"Created default security configuration: {config_file}")
                except Exception as e:
                    logger.error(f"Failed to create security configuration: {e}")
                    
            return success, issues
        
        # Check configuration file permissions
        try:
            config_stats = os.stat(config_file)
            config_mode = config_stats.st_mode & 0o777
            
            if config_mode != 0o600:
                issues.append(f"Security configuration file has incorrect permissions: {oct(config_mode)}, should be 0o600")
                success = False
                
                # Try to fix
                if fix:
                    try:
                        os.chmod(config_file, 0o600)
                        logger.info(f"Fixed permissions on security configuration file: {config_file}")
                    except Exception as e:
                        logger.error(f"Failed to fix permissions on security configuration file: {e}")
        except Exception as e:
            issues.append(f"Could not check security configuration file permissions: {e}")
            success = False
        
        # Load and validate configuration
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            
            # Check required sections
            required_sections = ["logging", "file_security", "api_security", "database"]
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing required section in security configuration: {section}")
                    success = False
            
            # Check logging configuration
            if "logging" in config:
                logging_config = config["logging"]
                if "level" not in logging_config:
                    issues.append("Missing log level in logging configuration")
                    success = False
                if "mask_patterns" not in logging_config:
                    issues.append("Missing mask patterns in logging configuration")
                    success = False
            
            # Check file security configuration
            if "file_security" in config:
                file_config = config["file_security"]
                if "temp_directory" not in file_config:
                    issues.append("Missing temporary directory in file security configuration")
                    success = False
                if "allowed_extensions" not in file_config:
                    issues.append("Missing allowed extensions in file security configuration")
                    success = False
            
            # Check API security configuration
            if "api_security" in config:
                api_config = config["api_security"]
                if "rate_limit_enabled" not in api_config:
                    issues.append("Missing rate limit enabled flag in API security configuration")
                    success = False
            
            # Check database configuration
            if "database" in config:
                db_config = config["database"]
                if "encryption_enabled" not in db_config:
                    issues.append("Missing encryption enabled flag in database configuration")
                    success = False
                if "backup_enabled" not in db_config:
                    issues.append("Missing backup enabled flag in database configuration")
                    success = False
        
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON in security configuration file: {e}")
            success = False
        except Exception as e:
            issues.append(f"Error validating security configuration: {e}")
            success = False
    
    except Exception as e:
        issues.append(f"Error verifying security configuration: {e}")
        success = False
    
    if verbose:
        if success:
            logger.info("Security configuration verification passed")
        else:
            logger.warning("Security configuration verification failed")
            for issue in issues:
                logger.warning(f"- {issue}")
    
    return success, issues


def verify_api_keys(verbose=False):
    """Verify API keys.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (success, issues)
    """
    issues = []
    success = True
    
    try:
        # Get credential manager
        credential_manager = get_credential_manager()
        
        # Check for API keys
        anthropic_key = credential_manager.get_credential(CredentialManager.ANTHROPIC_API_KEY)
        openai_key = credential_manager.get_credential(CredentialManager.OPENAI_API_KEY)
        google_key = credential_manager.get_credential(CredentialManager.GOOGLE_API_KEY)
        
        api_keys = {
            "Anthropic": anthropic_key,
            "OpenAI": openai_key,
            "Google": google_key
        }
        
        # Check each API key
        for provider, key in api_keys.items():
            if not key:
                issues.append(f"Missing {provider} API key")
                success = False
            elif len(key) < 8:
                issues.append(f"{provider} API key is too short")
                success = False
            elif verbose:
                logger.info(f"{provider} API key is present")
    
    except Exception as e:
        issues.append(f"Error verifying API keys: {e}")
        success = False
    
    if verbose:
        if success:
            logger.info("API key verification passed")
        else:
            logger.warning("API key verification failed")
            for issue in issues:
                logger.warning(f"- {issue}")
    
    return success, issues


def verify_database_security(verbose=False):
    """Verify database security.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (success, issues)
    """
    issues = []
    success = True
    
    try:
        # Check database file
        db_path = "./.cache/workflow.db"
        db_file = Path(db_path)
        
        if not db_file.exists():
            if verbose:
                logger.info(f"Database file does not exist: {db_file}")
            return success, issues
        
        # Check database file permissions
        try:
            db_stats = os.stat(db_file)
            db_mode = db_stats.st_mode & 0o777
            
            if db_mode > 0o600:
                issues.append(f"Database file has insecure permissions: {oct(db_mode)}, should be 0o600 or less")
                success = False
        except Exception as e:
            issues.append(f"Could not check database file permissions: {e}")
            success = False
        
        # Verify database connection and pragmas
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check pragmas
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            if journal_mode != "wal":
                issues.append(f"Database journal mode is not WAL: {journal_mode}")
                success = False
            
            cursor.execute("PRAGMA foreign_keys;")
            foreign_keys = cursor.fetchone()[0]
            if foreign_keys != 1:
                issues.append("Database foreign keys are not enabled")
                success = False
            
            cursor.execute("PRAGMA secure_delete;")
            secure_delete = cursor.fetchone()[0]
            if secure_delete != 1:
                issues.append("Database secure delete is not enabled")
                success = False
            
            conn.close()
        except Exception as e:
            issues.append(f"Error checking database pragmas: {e}")
            success = False
    
    except Exception as e:
        issues.append(f"Error verifying database security: {e}")
        success = False
    
    if verbose:
        if success:
            logger.info("Database security verification passed")
        else:
            logger.warning("Database security verification failed")
            for issue in issues:
                logger.warning(f"- {issue}")
    
    return success, issues


def verify_module_installation(verbose=False):
    """Verify security-related module installation.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (success, issues)
    """
    issues = []
    success = True
    
    # List of required security modules
    required_modules = [
        "cryptography",
        "bcrypt",
        "pyjwt",
        "pyopenssl"
    ]
    
    missing_modules = []
    
    # Check each module
    for module in required_modules:
        try:
            __import__(module)
            if verbose:
                logger.info(f"Module '{module}' is installed")
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        issues.append(f"Missing security modules: {', '.join(missing_modules)}")
        issues.append("Install them with: pip install " + " ".join(missing_modules))
        success = False
    
    if verbose:
        if success:
            logger.info("Module installation verification passed")
        else:
            logger.warning("Module installation verification failed")
            for issue in issues:
                logger.warning(f"- {issue}")
    
    return success, issues


def main():
    """Main function for security verification."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    all_success = True
    all_issues = []
    
    logger.info("Starting security verification")
    
    # Verify credential manager
    success, issues = verify_credential_manager(args.config_dir, args.verbose, args.fix)
    all_success = all_success and success
    all_issues.extend(issues)
    
    # Verify security configuration
    success, issues = verify_security_config(args.config_dir, args.verbose, args.fix)
    all_success = all_success and success
    all_issues.extend(issues)
    
    # Verify API keys
    success, issues = verify_api_keys(args.verbose)
    all_success = all_success and success
    all_issues.extend(issues)
    
    # Verify database security
    success, issues = verify_database_security(args.verbose)
    all_success = all_success and success
    all_issues.extend(issues)
    
    # Verify module installation
    success, issues = verify_module_installation(args.verbose)
    all_success = all_success and success
    all_issues.extend(issues)
    
    # Print summary
    if all_success:
        logger.info("Security verification passed! All checks completed successfully.")
        return 0
    else:
        logger.warning("Security verification failed! Please address the following issues:")
        for issue in all_issues:
            logger.warning(f"- {issue}")
        
        if args.fix:
            logger.info("Some issues may have been fixed automatically. Please run verification again.")
        else:
            logger.info("Run with --fix to attempt automatic fixes for these issues.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())