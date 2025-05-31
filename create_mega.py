#!/usr/bin/env python3

"""
Create Mega-Result Script

This script creates a mega-result by directly using the database and file system,
bypassing the interactive prompts in paper_revision.py.
"""

import os
import sys
import sqlite3
import datetime
import json
import random
import time
import subprocess
from colorama import Fore, Style, init

# Initialize colorama
init()

def generate_run_id():
    """Generate a unique run ID based on the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return timestamp

def create_mega_result_in_db(run_id, source_run_ids, provider="google", model="gemini-1.5-pro (powerful, 1M context)"):
    """Create a mega-result entry in the database."""
    # Connect to the database
    db_path = "./.cache/workflow.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the mega-result entry
    now = datetime.datetime.now().isoformat()
    settings = {
        "source_runs": source_run_ids,
        "mega_result": True
    }
    
    cursor.execute('''
    INSERT INTO runs (run_id, timestamp, provider, model, operation_mode, 
                     status, start_time, settings)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, now, provider, model, "final", 
         "started", now, json.dumps(settings)))
    
    # Copy files from source runs
    for source_run_id in source_run_ids:
        # Get files from the source run
        cursor.execute('SELECT * FROM files WHERE run_id = ?', (source_run_id,))
        files = cursor.fetchall()
        
        # Copy each file to the new run
        for file in files:
            if file[0] and file[1]:  # file_id and run_id
                # Generate a new file ID for the merged run
                new_file_id = file[0] + f"_merged_{run_id}"
                
                cursor.execute('''
                INSERT INTO files (file_id, run_id, original_path, processed_path, 
                                 file_type, size, token_estimate, page_count, processed_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (new_file_id, run_id, file[2], file[3], file[4], file[5], file[6], file[7], now))
    
    # Commit changes
    conn.commit()
    conn.close()
    
    return run_id

def get_latest_runs(count=3):
    """Get the latest runs from the database."""
    # Connect to the database
    db_path = "./.cache/workflow.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the latest runs with different providers
    cursor.execute('''
    SELECT run_id, provider, model, timestamp
    FROM runs
    WHERE status = 'completed'
    ORDER BY timestamp DESC
    LIMIT 20
    ''')
    
    all_runs = cursor.fetchall()
    conn.close()
    
    # Get one run per provider if possible
    providers_seen = set()
    selected_runs = []
    
    for run in all_runs:
        run_id, provider, model, timestamp = run
        if provider not in providers_seen and len(selected_runs) < count:
            selected_runs.append(run)
            providers_seen.add(provider)
    
    # If we don't have enough runs with different providers, just take the latest runs
    if len(selected_runs) < count:
        return all_runs[:count]
    
    return selected_runs

def main():
    """Main entry point."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}MEGA-RESULT CREATOR (NON-INTERACTIVE){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    # Get source runs
    print(f"\n{Fore.BLUE}[STEP 1]{Style.RESET_ALL} Finding latest runs to merge...")
    source_runs = get_latest_runs(3)
    
    if not source_runs:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No source runs found in the database")
        return
    
    source_run_ids = [run[0] for run in source_runs]
    
    # Print source runs
    print(f"Found {len(source_runs)} source runs:")
    for i, (run_id, provider, model, timestamp) in enumerate(source_runs, 1):
        print(f"  {i}. {provider.capitalize()} {model} (Run ID: {run_id})")
    
    # Create a mega-result
    print(f"\n{Fore.BLUE}[STEP 2]{Style.RESET_ALL} Creating mega-result in database...")
    run_id = generate_run_id()
    create_mega_result_in_db(run_id, source_run_ids)
    
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Created mega-result with run ID: {run_id}")
    
    # Generate the mega-result directory
    print(f"\n{Fore.BLUE}[STEP 3]{Style.RESET_ALL} Generating mega-result directory...")
    time.sleep(1)  # Short delay to ensure database writes are complete
    
    process = subprocess.Popen(
        f"python generate_mega_dir.py {run_id}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    output, error = process.communicate()
    print(output)
    
    if error:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {error}")
    
    # View the mega-result
    print(f"\n{Fore.BLUE}[STEP 4]{Style.RESET_ALL} Viewing mega-result...")
    
    process = subprocess.Popen(
        f"python paper_revision.py --view-mega {run_id}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    output, error = process.communicate()
    print(output)
    
    print(f"\n{Fore.GREEN}[COMPLETE]{Style.RESET_ALL} Mega-result creation process completed!")
    print(f"You can find all files in: ./tobe/MEGA/{run_id}")
    print(f"To view the mega-result again, run: python paper_revision.py --view-mega {run_id}")

if __name__ == "__main__":
    main()