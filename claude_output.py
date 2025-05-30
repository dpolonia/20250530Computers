#\!/usr/bin/env python3
"""
Claude Output Script

This script captures standard output from another command and formats it for Claude.
It can be used to pipe output directly to a file that can be sent to Claude.

Usage:
    python claude_output.py [--file FILENAME] COMMAND
    
Example:
    python claude_output.py --file claude_input.txt python paper_revision.py --mode training
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Capture command output for Claude")
    parser.add_argument("--file", help="Output file (default: claude_output_TIMESTAMP.txt)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute and capture")
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        print("Error: No command specified")
        parser.print_help()
        sys.exit(1)
    
    # Create output filename if not specified
    if not args.file:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.file = f"claude_output_{timestamp}.txt"
    
    # Execute the command and capture output
    try:
        # Join command arguments into a string
        cmd = " ".join(args.command)
        
        # Run the command and capture output
        print(f"Running command: {cmd}")
        print(f"Output will be saved to: {args.file}")
        
        # Open the output file
        with open(args.file, "w") as f:
            # Write header
            f.write(f"Command output from: {cmd}\n")
            f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write("-" * 80 + "\n\n")
            
            # Execute command and capture output in real-time
            process = subprocess.Popen(
                args.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Process output line by line
            for line in process.stdout:
                sys.stdout.write(line)  # Echo to console
                f.write(line)           # Write to file
                
            # Wait for process to complete
            return_code = process.wait()
            
            # Write footer
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"Command completed with return code: {return_code}\n")
        
        print(f"\nOutput saved to {args.file}")
        
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
