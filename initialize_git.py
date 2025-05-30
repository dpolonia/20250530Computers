#!/usr/bin/env python3
"""
Initialize Git repository and push to GitHub.

This script creates a new Git repository, adds all files,
creates an initial commit, and pushes to GitHub.
"""

import os
import sys
import subprocess
import getpass
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

def run_command(command, error_message=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            shell=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if error_message:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {error_message}")
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Command: {command}")
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Error: {e.stderr}")
        return None

def initialize_git():
    """Initialize Git repository."""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Initializing Git repository...")
    
    # Check if git is already initialized
    if os.path.exists(".git"):
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Git repository already exists.")
        return True
    
    # Initialize git repository
    if not run_command("git init", "Failed to initialize Git repository"):
        return False
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Creating .gitignore file...")
        with open(".gitignore", "w") as f:
            f.write("# Python cache files\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n\n")
            f.write("# Environment variables\n")
            f.write(".env\n\n")
            f.write("# Virtual environment\n")
            f.write("venv/\n")
            f.write("env/\n")
            f.write("ENV/\n\n")
            f.write("# Output files in tobe directory with timestamps\n")
            f.write("tobe/90*.docx\n")
            f.write("tobe/91*.docx\n")
            f.write("tobe/92*.docx\n")
            f.write("tobe/93*.docx\n")
            f.write("tobe/94*.docx\n")
            f.write("tobe/zz*.bib\n\n")
            f.write("# Logs\n")
            f.write("*.log\n\n")
            f.write("# IDE files\n")
            f.write(".vscode/\n")
            f.write(".idea/\n")
            f.write("*.swp\n")
            f.write("*.swo\n\n")
            f.write("# Distribution / packaging\n")
            f.write("dist/\n")
            f.write("build/\n")
            f.write("*.egg-info/\n")
    
    # Add all files
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Adding files to Git...")
    if not run_command("git add .", "Failed to add files to Git"):
        return False
    
    # Create initial commit
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Creating initial commit...")
    if not run_command(
        'git commit -m "Initial commit: Paper revision tool"',
        "Failed to create initial commit"
    ):
        return False
    
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Git repository initialized successfully.")
    return True

def push_to_github(username, password):
    """Push to GitHub repository."""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Pushing to GitHub...")
    
    # Add remote if it doesn't exist
    remotes = run_command("git remote", "Failed to list remotes")
    if "origin" not in remotes.split():
        # Add origin remote
        repo_url = f"https://{username}:{password}@github.com/{username}/20250530Computers.git"
        if not run_command(
            f'git remote add origin "{repo_url}"',
            "Failed to add GitHub remote"
        ):
            return False
    
    # Push to GitHub
    if not run_command("git push -u origin master", "Failed to push to GitHub"):
        return False
    
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Successfully pushed to GitHub.")
    return True

def main():
    """Main function."""
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}GitHub Repository Setup{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    # Initialize Git
    if not initialize_git():
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to initialize Git repository.")
        return
    
    # Ask for GitHub credentials
    print("\nPlease enter your GitHub credentials:")
    username = input("Username (dpolonia): ") or "dpolonia"
    password = getpass.getpass("Password: ")
    
    # Push to GitHub
    if not push_to_github(username, password):
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to push to GitHub.")
        return
    
    print(f"\n{Fore.GREEN}Repository setup complete!{Style.RESET_ALL}")
    print(f"GitHub URL: https://github.com/{username}/20250530Computers")

if __name__ == "__main__":
    main()