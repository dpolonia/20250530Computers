#!/usr/bin/env python3
"""
Test environment setup for the paper revision tool.

This script checks if all required dependencies are installed
and if the API keys are properly configured.
"""

import os
import sys
import importlib
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

def check_module(module_name):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} {module_name}")
        return True
    except ImportError:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} {module_name}")
        return False

def check_api_keys():
    """Check if API keys are configured and validate them."""
    load_dotenv()
    
    keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    has_keys = False
    print(f"\n{Fore.CYAN}Checking API Keys:{Style.RESET_ALL}")
    
    for key_name, key_value in keys.items():
        if not key_value:
            print(f"{Fore.YELLOW}[!]{Style.RESET_ALL} {key_name} not found")
            continue
            
        # Key exists, now validate it
        print(f"{Fore.BLUE}[*]{Style.RESET_ALL} Testing {key_name}...", end="", flush=True)
        try:
            if key_name == "ANTHROPIC_API_KEY":
                import anthropic
                client = anthropic.Anthropic(api_key=key_value)
                # Just list models to verify the key works
                client.models.list()
                print(f"\r{Fore.GREEN}[✓]{Style.RESET_ALL} {key_name} is valid and working   ")
                has_keys = True
            elif key_name == "OPENAI_API_KEY":
                import openai
                client = openai.OpenAI(api_key=key_value)
                # Just list models to verify the key works
                client.models.list()
                print(f"\r{Fore.GREEN}[✓]{Style.RESET_ALL} {key_name} is valid and working   ")
                has_keys = True
            elif key_name == "GOOGLE_API_KEY":
                import google.generativeai as genai
                genai.configure(api_key=key_value)
                # Just list models to verify the key works
                models = genai.list_models()
                list(models)  # Force evaluation of generator
                print(f"\r{Fore.GREEN}[✓]{Style.RESET_ALL} {key_name} is valid and working   ")
                has_keys = True
        except Exception as e:
            print(f"\r{Fore.RED}[✗]{Style.RESET_ALL} {key_name} is invalid or expired: {str(e)}")
    
    return has_keys

def check_directories():
    """Check if required directories exist."""
    directories = ["asis", "tobe", "src"]
    
    print(f"\n{Fore.CYAN}Checking Directories:{Style.RESET_ALL}")
    
    all_exist = True
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} {directory}/")
        else:
            print(f"{Fore.RED}[✗]{Style.RESET_ALL} {directory}/")
            all_exist = False
    
    return all_exist

def check_input_files():
    """Check if required input files exist."""
    input_files = [
        "asis/00.pdf",
        "asis/01.pdf",
        "asis/02.pdf",
        "asis/03.pdf",
        "asis/04.pdf",
        "asis/05.pdf",
        "asis/06.pdf",
        "asis/07.pdf",
        "asis/08.pdf",
        "asis/zz.bib"
    ]
    
    print(f"\n{Fore.CYAN}Checking Input Files:{Style.RESET_ALL}")
    
    all_exist = True
    for file_path in input_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} {file_path}")
        else:
            print(f"{Fore.RED}[✗]{Style.RESET_ALL} {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Main function."""
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Paper Revision Tool - Environment Test{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"\n{Fore.CYAN}Python Version:{Style.RESET_ALL} {python_version}")
    
    # Check required modules
    required_modules = [
        "dotenv",
        "anthropic",
        "openai",
        "google.generativeai",
        "fitz",  # PyMuPDF
        "docx",
        "bibtexparser",
        "requests",
        "tqdm",
        "colorama",
        "click"
    ]
    
    print(f"\n{Fore.CYAN}Checking Required Modules:{Style.RESET_ALL}")
    all_modules_installed = all(check_module(module) for module in required_modules)
    
    # Check API keys
    has_api_keys = check_api_keys()
    
    # Check directories
    has_directories = check_directories()
    
    # Check input files
    has_input_files = check_input_files()
    
    # Print summary
    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Summary:{Style.RESET_ALL}")
    
    if all_modules_installed:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} All required modules are installed")
    else:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} Some required modules are missing")
    
    if has_api_keys:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} At least one API key is validated and working")
    else:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} No valid API keys found. Please check your .env file and ensure keys are not expired")
    
    if has_directories:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} All required directories exist")
    else:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} Some required directories are missing")
    
    if has_input_files:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} All required input files exist")
    else:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} Some required input files are missing")
    
    # Final verdict
    if all([all_modules_installed, has_api_keys, has_directories, has_input_files]):
        print(f"\n{Fore.GREEN}Environment is correctly set up!{Style.RESET_ALL}")
        print(f"You can run the paper revision tool with: python paper_revision.py")
    else:
        print(f"\n{Fore.YELLOW}Environment is not completely set up.{Style.RESET_ALL}")
        print(f"Please fix the issues above before running the paper revision tool.")

def test_api_keys_only():
    """Run only the API key validation tests."""
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Paper Revision Tool - API Key Validation{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    has_api_keys = check_api_keys()
    
    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}API Key Validation Summary:{Style.RESET_ALL}")
    
    if has_api_keys:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} At least one API key is validated and working")
    else:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} No valid API keys found. Please check your .env file and ensure keys are not expired")
    
    print(f"\n{Fore.CYAN}Note: You need at least one valid API key to use the paper revision tool.{Style.RESET_ALL}")

if __name__ == "__main__":
    # Check if we should only test API keys
    if len(sys.argv) > 1 and sys.argv[1] == "--api-keys":
        test_api_keys_only()
    else:
        main()