#!/usr/bin/env python3

"""
Mega-Run Script

This script runs the paper_revision.py with mega-result creation and automatically
generates the mega-result directory with all output files.
"""

import os
import sys
import subprocess
import re
import time
from colorama import Fore, Style, init

# Initialize colorama
init()

def extract_run_id(output):
    """Extract the run ID from the output of paper_revision.py."""
    # Match pattern like "You can find the merged results in the database using run ID: 20250531043930"
    match = re.search(r"run ID: (\d+)", output)
    if match:
        return match.group(1)
    return None

def main():
    """Main entry point."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}MEGA-RESULT AUTOMATIC RUNNER{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    # Run paper_revision.py to create the mega-result
    print(f"\n{Fore.BLUE}[STEP 1]{Style.RESET_ALL} Running paper_revision.py to create mega-result entry...")
    
    # Create a temporary file with the inputs
    with open("temp_inputs.txt", "w") as f:
        f.write("3\nY\n3\nY\n")  # Mode 3, Yes to recommendation, Provider 3 (Google), Yes to model
    
    try:
        # Run the command with input from the file
        cmd = "python paper_revision.py < temp_inputs.txt"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()
        
        # Remove the temporary file
        os.remove("temp_inputs.txt")
        
        # Print the output
        print(output)
        
        if error:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {error}")
            return
        
        # Extract the run ID from the output
        run_id = extract_run_id(output)
        
        if not run_id:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not extract run ID from output")
            return
        
        print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Created mega-result with run ID: {run_id}")
        
        # Run generate_mega_dir.py to create the mega-result directory
        print(f"\n{Fore.BLUE}[STEP 2]{Style.RESET_ALL} Generating mega-result directory...")
        
        # Allow a short delay to ensure database writes are complete
        time.sleep(1)
        
        # Run the generate_mega_dir.py script
        generate_cmd = f"python generate_mega_dir.py {run_id}"
        process = subprocess.Popen(generate_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        gen_output, gen_error = process.communicate()
        
        # Print the output
        print(gen_output)
        
        if gen_error:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {gen_error}")
            return
        
        # Run paper_revision.py --view-mega to show the result
        print(f"\n{Fore.BLUE}[STEP 3]{Style.RESET_ALL} Viewing mega-result...")
        
        view_cmd = f"python paper_revision.py --view-mega {run_id}"
        process = subprocess.Popen(view_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        view_output, view_error = process.communicate()
        
        # Print the output
        print(view_output)
        
        if view_error:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {view_error}")
            return
        
        print(f"\n{Fore.GREEN}[COMPLETE]{Style.RESET_ALL} Mega-result creation and directory generation complete!")
        print(f"You can find all files in: ./tobe/MEGA/{run_id}")
        
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} An error occurred: {str(e)}")
        # Clean up the temporary file if it exists
        if os.path.exists("temp_inputs.txt"):
            os.remove("temp_inputs.txt")

if __name__ == "__main__":
    main()