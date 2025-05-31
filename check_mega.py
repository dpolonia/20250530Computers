#!/usr/bin/env python3

import os
import sqlite3
import json
import sys
from pprint import pprint

# Create database connection
db_path = "./.cache/workflow.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Get run information
run_id = "20250531041607"
if len(sys.argv) > 1:
    run_id = sys.argv[1]

cursor = conn.cursor()

# Print run information
cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
run = cursor.fetchone()

if not run:
    print(f"Run with ID {run_id} not found")
    sys.exit(1)

print("=" * 60)
print(f"Run ID: {run_id}")
print(f"Provider: {run['provider']}")
print(f"Model: {run['model']}")
print(f"Operation mode: {run['operation_mode']}")
print(f"Status: {run['status']}")
print(f"Start time: {run['start_time']}")
print(f"End time: {run['end_time']}")
print("=" * 60)

# Get files from the database
print("\nFILES:")
cursor.execute("SELECT * FROM files WHERE run_id = ?", (run_id,))
files = cursor.fetchall()
for file in files:
    print(f"  - {file['file_type']}: {file['processed_path']} (Original: {file['original_path']})")

# Get steps from the database
print("\nSTEPS:")
cursor.execute("SELECT * FROM steps WHERE run_id = ? ORDER BY step_number", (run_id,))
steps = cursor.fetchall()
for step in steps:
    print(f"  - [{step['status']}] Step {step['step_number']}: {step['step_name']}")
    if step['output_file']:
        print(f"    Output: {step['output_file']}")

# Check what output files we can find
print("\nOUTPUT FILES CHECK:")
cursor.execute("SELECT step_name, output_file FROM steps WHERE run_id = ? AND output_file IS NOT NULL", (run_id,))
step_files = cursor.fetchall()
for step_file in step_files:
    exists = os.path.exists(step_file['output_file'])
    print(f"  - {step_file['step_name']}: {step_file['output_file']} - {'EXISTS' if exists else 'MISSING'}")

# Check mega directory structure
mega_dir = f"./tobe/MEGA/{run_id}"
print(f"\nMEGA DIRECTORY ({mega_dir}):")
if os.path.exists(mega_dir):
    # List root directory
    print("  Root directory:")
    root_files = [f for f in os.listdir(mega_dir) if os.path.isfile(os.path.join(mega_dir, f))]
    for file in root_files:
        print(f"    - {file}")
    
    # List papers directory
    papers_dir = os.path.join(mega_dir, "papers")
    if os.path.exists(papers_dir):
        print("  Papers directory:")
        papers_files = [f for f in os.listdir(papers_dir) if os.path.isfile(os.path.join(papers_dir, f))]
        if papers_files:
            for file in papers_files:
                print(f"    - {file}")
        else:
            print("    (empty)")
    
    # List reports directory
    reports_dir = os.path.join(mega_dir, "reports")
    if os.path.exists(reports_dir):
        print("  Reports directory:")
        reports_files = [f for f in os.listdir(reports_dir) if os.path.isfile(os.path.join(reports_dir, f))]
        if reports_files:
            for file in reports_files:
                print(f"    - {file}")
        else:
            print("    (empty)")
            
    # List metadata directory
    metadata_dir = os.path.join(mega_dir, "metadata")
    if os.path.exists(metadata_dir):
        print("  Metadata directory:")
        metadata_files = [f for f in os.listdir(metadata_dir) if os.path.isfile(os.path.join(metadata_dir, f))]
        if metadata_files:
            for file in metadata_files:
                print(f"    - {file}")
        else:
            print("    (empty)")
else:
    print(f"  Directory not found")

# Close connection
conn.close()