#!/usr/bin/env python3

import os
import sqlite3
import json
import sys
import datetime
import shutil
import random
from pprint import pprint

# Create a test mega-result with sample files
run_id = f"2025053199{random.randint(1000, 9999)}"  # Test run ID with random suffix
provider = "test"
model = "test-model"
operation_mode = "mega"

# Create database connection
db_path = "./.cache/workflow.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Create temporary output files for testing
output_dir = "./tobe/test_mega_output"
os.makedirs(output_dir, exist_ok=True)

# Create test files
files_to_create = [
    {"name": "90revised_paper.docx", "type": "revised_paper"},
    {"name": "91revision_summary.docx", "type": "revision_summary"},
    {"name": "92editor_letter.docx", "type": "editor_letter"},
    {"name": "93changes_document.docx", "type": "changes_document"},
    {"name": "94assessment.docx", "type": "assessment"},
    {"name": "95bibliography.bib", "type": "bibliography"},
    {"name": "96log.txt", "type": "log"},
    {"name": "97cost_report.txt", "type": "cost_report"}
]

# Create files with content
file_paths = {}
for file_info in files_to_create:
    file_path = os.path.join(output_dir, file_info["name"])
    file_type = file_info["type"]
    
    # Create a simple text file with the type as content
    with open(file_path, 'w') as f:
        f.write(f"This is a test {file_type} file for mega-result testing.\n")
        f.write("It would normally contain actual content from the paper revision process.\n")
        f.write(f"File type: {file_type}")
    
    file_paths[file_type] = file_path
    print(f"Created test file: {file_path}")

# Insert run into database
now = datetime.datetime.now().isoformat()
cursor.execute('''
INSERT INTO runs (run_id, timestamp, provider, model, operation_mode, 
                status, start_time, end_time, total_tokens, total_cost,
                evaluation_tokens, evaluation_cost, settings)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (run_id, now, provider, model, operation_mode, 
     "completed", now, now, 50000, 1.25, 
     5000, 0.25, json.dumps({"test": True})))

# Insert steps with output files
step_id = 1
for file_info in files_to_create:
    file_type = file_info["type"]
    file_path = file_paths[file_type]
    
    cursor.execute('''
    INSERT INTO steps (run_id, step_number, step_name, start_time, end_time,
                      duration, status, output_file)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, step_id, f"Create {file_type}", now, now, 
         10.5, "completed", file_path))
    
    step_id += 1
    
# Commit changes
conn.commit()

print(f"\nCreated test mega-result with ID: {run_id}")
print("You can now test the mega-result viewing functionality with:")
print(f"python paper_revision.py --view-mega {run_id}")

# Close connection
conn.close()