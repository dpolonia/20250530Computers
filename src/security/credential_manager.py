"""
Secure credential management for API keys and sensitive information.

This module handles the secure storage, retrieval, and validation of API keys
and other sensitive credentials used throughout the system.
"""

import os
import json
import logging
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Manages secure storage and retrieval of API keys and other credentials.
    
    Features:
    - Secure storage of API keys using encryption
    - Validation of API keys before use
    - Masked logging to prevent credential exposure
    - Support for key rotation
    """
    
    # Constants for credential types
    ANTHROPIC_API_KEY = "anthropic_api_key"
    OPENAI_API_KEY = "openai_api_key"
    GOOGLE_API_KEY = "google_api_key"
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the credential manager.
        
        Args:
            config_dir: Directory to store encrypted credentials
                        Defaults to ~/.paper_revision_secure/
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".paper_revision_secure"
        else:
            self.config_dir = Path(config_dir)
            
        # Ensure the config directory exists with proper permissions
        self._ensure_secure_directory()
        
        # Initialize encryption key
        self._encryption_key = self._get_or_create_encryption_key()
        
        # Cache for credentials to avoid frequent decryption
        self._credential_cache = {}
        
    def _ensure_secure_directory(self) -> None:
        """
        Ensure the config directory exists with proper permissions.
        Directory permissions are set to 700 (rwx------)
        """
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
            
        # Set secure permissions (rwx------ / 700)
        try:
            os.chmod(self.config_dir, 0o700)
        except Exception as e:
            logger.warning(f"Could not set secure permissions on {self.config_dir}: {e}")
            
    def _get_or_create_encryption_key(self) -> bytes:
        """
        Get or create an encryption key for securing credentials.
        
        Returns:
            Fernet encryption key as bytes
        """
        key_file = self.config_dir / "encryption.key"
        
        if key_file.exists():
            # Read existing key
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            # Generate a new key
            salt = os.urandom(16)
            machine_id = self._get_machine_identifier()
            
            # Derive a key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
            
            # Save the key with secure permissions
            with open(key_file, "wb") as f:
                f.write(key)
            
            os.chmod(key_file, 0o600)  # rw-------
            
        return key
    
    def _get_machine_identifier(self) -> str:
        """
        Get a unique identifier for the machine to derive encryption key.
        This helps tie the encryption to the specific machine.
        
        Returns:
            A string identifier unique to this machine
        """
        # Try multiple sources to create a unique identifier
        identifiers = []
        
        # Try /etc/machine-id
        try:
            with open("/etc/machine-id", "r") as f:
                identifiers.append(f.read().strip())
        except Exception:
            pass
            
        # Try hostname
        try:
            identifiers.append(os.uname().nodename)
        except Exception:
            pass
        
        # Combine identifiers with a salt
        combined = "paper_revision_" + "_".join(identifiers) + "_salt_9a7b3c"
        
        # Hash the combined identifier
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def store_credential(self, 
                        credential_type: str, 
                        value: str, 
                        overwrite: bool = False) -> bool:
        """
        Securely store a credential.
        
        Args:
            credential_type: Type of credential (use class constants)
            value: The credential value to store
            overwrite: Whether to overwrite existing credential
            
        Returns:
            True if successful, False otherwise
        """
        if not value or not credential_type:
            logger.error("Cannot store empty credential")
            return False
            
        credential_file = self.config_dir / f"{credential_type}.enc"
        
        if credential_file.exists() and not overwrite:
            logger.warning(f"Credential {credential_type} already exists. Use overwrite=True to replace.")
            return False
            
        # Encrypt the credential
        fernet = Fernet(self._encryption_key)
        encrypted_value = fernet.encrypt(value.encode())
        
        # Store the encrypted credential
        with open(credential_file, "wb") as f:
            f.write(encrypted_value)
            
        # Set secure permissions
        os.chmod(credential_file, 0o600)  # rw-------
        
        # Update cache
        self._credential_cache[credential_type] = value
        
        logger.info(f"Stored credential: {credential_type}")
        return True
    
    def get_credential(self, 
                      credential_type: str, 
                      env_var: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a credential, first checking encrypted storage,
        then environment variables if specified.
        
        Args:
            credential_type: Type of credential to retrieve
            env_var: Environment variable name to check as fallback
            
        Returns:
            The credential value or None if not found
        """
        # Check cache first
        if credential_type in self._credential_cache:
            return self._credential_cache[credential_type]
            
        # Check encrypted storage
        credential_file = self.config_dir / f"{credential_type}.enc"
        
        if credential_file.exists():
            try:
                with open(credential_file, "rb") as f:
                    encrypted_value = f.read()
                    
                fernet = Fernet(self._encryption_key)
                decrypted_value = fernet.decrypt(encrypted_value).decode()
                
                # Update cache
                self._credential_cache[credential_type] = decrypted_value
                
                return decrypted_value
            except Exception as e:
                logger.error(f"Error decrypting credential {credential_type}: {e}")
                
        # Fall back to environment variable if specified
        if env_var:
            env_value = os.getenv(env_var)
            if env_value:
                logger.info(f"Using credential from environment variable: {self._mask_credential(env_var)}")
                
                # Update cache
                self._credential_cache[credential_type] = env_value
                
                return env_value
                
        logger.warning(f"Credential not found: {credential_type}")
        return None
    
    def validate_credential(self, 
                          credential_type: str, 
                          validation_fn: callable) -> bool:
        """
        Validate a credential using the provided validation function.
        
        Args:
            credential_type: Type of credential to validate
            validation_fn: Function that takes credential and returns bool
            
        Returns:
            True if credential is valid, False otherwise
        """
        credential = self.get_credential(credential_type)
        
        if not credential:
            logger.error(f"No credential found for {credential_type}")
            return False
            
        try:
            return validation_fn(credential)
        except Exception as e:
            logger.error(f"Error validating credential {credential_type}: {e}")
            return False
    
    def rotate_credential(self, 
                         credential_type: str, 
                         new_value: str) -> bool:
        """
        Rotate a credential by replacing it with a new value.
        
        Args:
            credential_type: Type of credential to rotate
            new_value: New credential value
            
        Returns:
            True if successful, False otherwise
        """
        # Store backup of old credential
        old_credential = self.get_credential(credential_type)
        
        if old_credential:
            backup_file = self.config_dir / f"{credential_type}.backup"
            
            # Encrypt the backup
            fernet = Fernet(self._encryption_key)
            encrypted_backup = fernet.encrypt(old_credential.encode())
            
            with open(backup_file, "wb") as f:
                f.write(encrypted_backup)
                
            os.chmod(backup_file, 0o600)  # rw-------
        
        # Store the new credential
        return self.store_credential(credential_type, new_value, overwrite=True)
    
    def _mask_credential(self, value: str) -> str:
        """
        Mask a credential for safe logging.
        
        Args:
            value: Credential to mask
            
        Returns:
            Masked credential string
        """
        if not value:
            return ""
            
        if len(value) <= 8:
            return "****"
            
        return value[:4] + "****" + value[-4:]
    
    def import_from_env(self, 
                       mapping: Dict[str, str], 
                       overwrite: bool = False) -> Dict[str, bool]:
        """
        Import credentials from environment variables.
        
        Args:
            mapping: Dictionary mapping credential types to env var names
            overwrite: Whether to overwrite existing credentials
            
        Returns:
            Dictionary of credential types to success status
        """
        results = {}
        
        for cred_type, env_var in mapping.items():
            env_value = os.getenv(env_var)
            
            if env_value:
                results[cred_type] = self.store_credential(
                    cred_type, env_value, overwrite=overwrite
                )
            else:
                results[cred_type] = False
                logger.warning(f"Environment variable not found: {env_var}")
                
        return results
    
    def export_to_env(self, credential_types: list) -> None:
        """
        Export credentials to environment variables.
        
        Args:
            credential_types: List of credential types to export
        """
        for cred_type in credential_types:
            value = self.get_credential(cred_type)
            
            if value:
                if cred_type == self.ANTHROPIC_API_KEY:
                    os.environ["ANTHROPIC_API_KEY"] = value
                elif cred_type == self.OPENAI_API_KEY:
                    os.environ["OPENAI_API_KEY"] = value
                elif cred_type == self.GOOGLE_API_KEY:
                    os.environ["GOOGLE_API_KEY"] = value
                    
                logger.info(f"Exported credential to environment: {cred_type}")
                
    def clear_credential(self, credential_type: str) -> bool:
        """
        Remove a credential from storage.
        
        Args:
            credential_type: Type of credential to clear
            
        Returns:
            True if successful, False otherwise
        """
        credential_file = self.config_dir / f"{credential_type}.enc"
        
        if credential_file.exists():
            try:
                credential_file.unlink()  # Delete the file
                
                # Remove from cache
                if credential_type in self._credential_cache:
                    del self._credential_cache[credential_type]
                    
                logger.info(f"Cleared credential: {credential_type}")
                return True
            except Exception as e:
                logger.error(f"Error clearing credential {credential_type}: {e}")
                
        return False


# Singleton instance for global use
_credential_manager = None

def get_credential_manager() -> CredentialManager:
    """
    Get the singleton credential manager instance.
    
    Returns:
        CredentialManager instance
    """
    global _credential_manager
    
    if _credential_manager is None:
        _credential_manager = CredentialManager()
        
    return _credential_manager