#!/usr/bin/env python3

"""
Generate Mega-Result Directory

This script takes a mega-result run ID and creates a proper directory structure
with all the output files from the source runs.
"""

import os
import sys
import sqlite3
import shutil
import json
from typing import List, Dict, Any
from colorama import Fore, Style

def init_colorama():
    """Initialize colorama for colored console output."""
    try:
        from colorama import init
        init()
    except ImportError:
        pass

def get_source_run_ids(cursor, mega_run_id):
    """Get the source run IDs for a mega-result."""
    # Try to get from settings
    cursor.execute("SELECT settings FROM runs WHERE run_id = ?", (mega_run_id,))
    run = cursor.fetchone()
    
    if not run:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Mega-result with run ID {mega_run_id} not found")
        return None
    
    try:
        settings = json.loads(run[0])
        if 'source_runs' in settings:
            return settings['source_runs']
    except:
        pass
    
    # Fallback: Use a database query to find runs created just before this one
    cursor.execute("""
    SELECT run_id FROM runs 
    WHERE timestamp < (SELECT timestamp FROM runs WHERE run_id = ?)
    ORDER BY timestamp DESC
    LIMIT 3
    """, (mega_run_id,))
    
    source_runs = [row[0] for row in cursor.fetchall()]
    if source_runs:
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Using heuristic to find source runs for {mega_run_id}")
        return source_runs
    
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not determine source runs for {mega_run_id}")
    return None

def find_output_files(cursor, run_id):
    """Find all output files for a run."""
    output_files = []
    
    # Get output files from steps
    cursor.execute("""
    SELECT step_name, output_file 
    FROM steps 
    WHERE run_id = ? AND output_file IS NOT NULL
    """, (run_id,))
    
    for row in cursor.fetchall():
        step_name, output_file = row
        
        # Determine file type based on step name and extension
        file_type = "unknown"
        if "revised_paper" in step_name.lower():
            file_type = "revised_paper"
        elif "revision_summary" in step_name.lower():
            file_type = "revision_summary"
        elif "editor_letter" in step_name.lower() or "response" in step_name.lower():
            file_type = "editor_letter"
        elif "changes_document" in step_name.lower():
            file_type = "changes_document"
        elif "assessment" in step_name.lower():
            file_type = "assessment"
        elif "bibliography" in step_name.lower() or output_file.endswith(".bib"):
            file_type = "bibliography"
        elif "log" in step_name.lower() or output_file.endswith(".log") or output_file.endswith("log.txt"):
            file_type = "log"
        elif "cost" in step_name.lower() or "report" in step_name.lower():
            file_type = "cost_report"
        
        # Check if file exists
        if os.path.exists(output_file):
            output_files.append({
                "file_type": file_type,
                "file_path": output_file,
                "step_name": step_name
            })
    
    # Get run information to find directory
    cursor.execute("SELECT provider, model, operation_mode FROM runs WHERE run_id = ?", (run_id,))
    run_info = cursor.fetchone()
    
    if run_info:
        provider = run_info[0]
        model = run_info[1]
        operation_mode = run_info[2]
        
        # Try to find directory based on operation mode and run ID
        possible_directories = [
            f"./tobe/{operation_mode.upper()}/{provider.capitalize()}_{model.replace(' ', '_')}/{run_id}",
            f"./tobe/{operation_mode.upper()}/{provider}_{model.replace(' ', '_')}/{run_id}",
            f"./tobe/{operation_mode.upper()}/{run_id}",
        ]
        
        # For specific naming patterns like A03_claude_3_5_sonnet_20241022_(balanced)
        # Add hardcoded directories for known models
        if provider == "anthropic":
            possible_directories.append(f"./tobe/{operation_mode.upper()}/A03_claude_3_5_sonnet_20241022_(balanced)/{run_id}")
        elif provider == "openai":
            possible_directories.append(f"./tobe/{operation_mode.upper()}/B03_gpt_4o_(powerful)/{run_id}")
        elif provider == "google":
            possible_directories.append(f"./tobe/{operation_mode.upper()}/C05_gemini_1.5_pro_(powerful,_1M_context)/{run_id}")
        
        # Add fuzzy match for directories with model name
        for root, dirs, files in os.walk("./tobe"):
            for dir in dirs:
                if run_id in dir:
                    possible_directories.append(os.path.join(root, dir))
        
        # Look for output files in the directories
        found_dir = None
        for dir_path in possible_directories:
            if os.path.exists(dir_path):
                found_dir = dir_path
                print(f"  - Found directory for run {run_id}: {dir_path}")
                break
        
        if found_dir:
            # Look for files in the directory
            for filename in os.listdir(found_dir):
                file_path = os.path.join(found_dir, filename)
                
                if os.path.isfile(file_path):
                    # Determine file type based on filename and extension
                    file_type = "unknown"
                    if filename.startswith("90") and filename.endswith(".docx"):
                        file_type = "revised_paper"
                    elif filename.startswith("91") and filename.endswith(".docx"):
                        file_type = "revision_summary"
                    elif filename.startswith("92") and filename.endswith(".docx"):
                        file_type = "editor_letter"
                    elif filename.startswith("93") and filename.endswith(".docx"):
                        file_type = "changes_document"
                    elif filename.startswith("94") and filename.endswith(".docx"):
                        file_type = "assessment"
                    elif filename.endswith(".bib"):
                        file_type = "bibliography"
                    elif "log" in filename.lower() and not "error" in filename.lower():
                        file_type = "log"
                    elif "cost" in filename.lower() or "report" in filename.lower():
                        file_type = "cost_report"
                    elif filename.endswith(".docx") and "paper" in filename.lower():
                        file_type = "revised_paper"
                    elif filename.endswith(".docx") and "summary" in filename.lower():
                        file_type = "revision_summary"
                    elif filename.endswith(".docx") and "letter" in filename.lower():
                        file_type = "editor_letter"
                    elif filename.endswith(".docx") and "change" in filename.lower():
                        file_type = "changes_document"
                    elif filename.endswith(".docx") and "assess" in filename.lower():
                        file_type = "assessment"
                    
                    output_files.append({
                        "file_type": file_type,
                        "file_path": file_path,
                        "step_name": f"Found in directory: {filename}"
                    })
                    print(f"    - Found file: {filename} (type: {file_type})")
    
    return output_files

def copy_to_mega_dir(mega_run_id, source_runs_info, output_files):
    """Copy all output files to the mega-result directory."""
    # Create mega-result directory and subdirectories
    mega_dir = f"./tobe/MEGA/{mega_run_id}"
    os.makedirs(mega_dir, exist_ok=True)
    
    # Create subdirectories for organization
    papers_dir = os.path.join(mega_dir, "papers")
    reports_dir = os.path.join(mega_dir, "reports")
    metadata_dir = os.path.join(mega_dir, "metadata")
    
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Copy all output files
    file_count = 0
    files_by_type = {}
    
    for source_info in source_runs_info:
        source_run_id = source_info["run_id"]
        source_provider = source_info["provider"]
        source_model = source_info["model"]
        
        for file_info in source_info["files"]:
            file_type = file_info["file_type"]
            file_path = file_info["file_path"]
            
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            
            files_by_type[file_type].append({
                "path": file_path,
                "provider": source_provider,
                "model": source_model,
                "run_id": source_run_id
            })
    
    # Process files by type
    for file_type, files in files_by_type.items():
        # Determine target directory
        if file_type in ["revised_paper", "editor_letter", "bibliography"]:
            target_subdir = papers_dir
        elif file_type in ["revision_summary", "changes_document", "assessment"]:
            target_subdir = reports_dir
        else:
            target_subdir = metadata_dir
        
        # Copy each file with provider prefix
        for file_info in files:
            file_path = file_info["path"]
            provider = file_info["provider"]
            model_short = file_info["model"].split()[0]  # Get first word of model name
            
            if os.path.exists(file_path):
                # Generate new filename with provider prefix
                orig_filename = os.path.basename(file_path)
                file_ext = os.path.splitext(orig_filename)[1]
                
                if file_type == "revised_paper":
                    new_name = f"{provider}_{model_short}_revised_paper{file_ext}"
                elif file_type == "revision_summary":
                    new_name = f"{provider}_{model_short}_revision_summary{file_ext}"
                elif file_type == "editor_letter":
                    new_name = f"{provider}_{model_short}_editor_letter{file_ext}"
                elif file_type == "changes_document":
                    new_name = f"{provider}_{model_short}_changes_document{file_ext}"
                elif file_type == "assessment":
                    new_name = f"{provider}_{model_short}_assessment{file_ext}"
                elif file_type == "bibliography":
                    new_name = f"{provider}_{model_short}_bibliography{file_ext}"
                elif file_type == "log":
                    new_name = f"{provider}_{model_short}_log{file_ext}"
                elif file_type == "cost_report":
                    new_name = f"{provider}_{model_short}_cost_report{file_ext}"
                else:
                    new_name = f"{provider}_{model_short}_{orig_filename}"
                
                # Copy to subdirectory
                target_path = os.path.join(target_subdir, new_name)
                shutil.copy2(file_path, target_path)
                
                # For common files, also copy to root with standardized names
                if file_type in ["revised_paper", "revision_summary", "editor_letter", "changes_document"]:
                    root_path = os.path.join(mega_dir, f"{file_type}{file_ext}")
                    shutil.copy2(file_path, root_path)
                
                file_count += 1
                print(f"  - Copied {file_type} from {provider} to {target_path}")
    
    # Create README file
    create_readme(mega_dir, mega_run_id, source_runs_info, files_by_type)
    
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Created mega-result directory with {file_count} files: {mega_dir}")
    return mega_dir, file_count

def create_readme(mega_dir, mega_run_id, source_runs_info, files_by_type):
    """Create a README.txt file in the mega-result directory."""
    readme_path = os.path.join(mega_dir, "README.txt")
    
    with open(readme_path, "w") as f:
        f.write(f"MEGA-RESULT {mega_run_id}\n")
        f.write("=" * 70 + "\n\n")
        
        # Write meta information
        f.write("SOURCE RUNS:\n")
        f.write("-" * 30 + "\n")
        for source_info in source_runs_info:
            f.write(f"• {source_info['provider'].capitalize()} {source_info['model']}\n")
            f.write(f"  Run ID: {source_info['run_id']}\n")
            f.write(f"  Status: {source_info['status']}\n")
            f.write("\n")
        
        # Write file information
        f.write("FILES INCLUDED:\n")
        f.write("-" * 30 + "\n")
        
        for file_type in ["revised_paper", "revision_summary", "changes_document", "editor_letter", "assessment", "bibliography", "log", "cost_report"]:
            if file_type in files_by_type:
                count = len(files_by_type[file_type])
                f.write(f"{file_type.replace('_', ' ').title()}: {count} file(s)\n")
                
                # List the providers
                providers = sorted(set(f["provider"] for f in files_by_type[file_type]))
                f.write(f"  From: {', '.join(p.capitalize() for p in providers)}\n")
        
        # Description and instructions
        f.write("\nDESCRIPTION:\n")
        f.write("-" * 30 + "\n")
        f.write("This directory contains the merged results from multiple provider runs.\n")
        f.write("The mega-result combines the outputs from different AI models to provide\n")
        f.write("a comprehensive revision of your academic paper.\n\n")
        
        f.write("HOW TO USE THESE FILES:\n")
        f.write("1. Start with the revised_paper files to see the different versions\n")
        f.write("2. Review the revision_summary files to understand the changes made\n")
        f.write("3. Check the editor_letter files for responses to reviewers\n")
        f.write("4. Refer to changes_document files for detailed breakdowns of edits\n\n")
        
        f.write("DIRECTORY STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        f.write("Root Directory: Key files from the primary provider for convenience\n")
        f.write("papers/: Contains the revised papers, editor letters, and bibliographies\n")
        f.write("reports/: Contains revision summaries, change documents, and assessments\n")
        f.write("metadata/: Contains logs, cost reports, and other auxiliary files\n")
    
    print(f"  - Created README.txt with information about the mega-result")

def main():
    """Main entry point."""
    init_colorama()
    
    # Get mega-result run ID from command line
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <mega_run_id>")
        return
    
    mega_run_id = sys.argv[1]
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Generating mega-result directory for run ID: {mega_run_id}")
    
    # Connect to database
    db_path = "./.cache/workflow.db"
    if not os.path.exists(db_path):
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get mega-result information
    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (mega_run_id,))
    mega_run = cursor.fetchone()
    
    if not mega_run:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Mega-result with run ID {mega_run_id} not found")
        conn.close()
        return
    
    mega_run_dict = dict(mega_run)
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Found mega-result: {mega_run_dict['provider']} {mega_run_dict['model']}")
    
    # Get source run IDs
    source_run_ids = get_source_run_ids(cursor, mega_run_id)
    if not source_run_ids:
        conn.close()
        return
    
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Found {len(source_run_ids)} source runs")
    
    # Get information about source runs
    source_runs_info = []
    output_files = []
    
    for source_run_id in source_run_ids:
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (source_run_id,))
        source_run = cursor.fetchone()
        
        if source_run:
            source_run_dict = dict(source_run)
            
            # Find output files for this run
            run_output_files = find_output_files(cursor, source_run_id)
            
            source_info = {
                "run_id": source_run_id,
                "provider": source_run_dict["provider"],
                "model": source_run_dict["model"],
                "operation_mode": source_run_dict["operation_mode"],
                "status": source_run_dict["status"],
                "files": run_output_files
            }
            
            source_runs_info.append(source_info)
            output_files.extend(run_output_files)
            
            print(f"  - Source run: {source_run_dict['provider']} {source_run_dict['model']} ({len(run_output_files)} files)")
    
    # Copy files to mega-result directory
    mega_dir, file_count = copy_to_mega_dir(mega_run_id, source_runs_info, output_files)
    
    # Set run status to completed
    cursor.execute("""
    UPDATE runs 
    SET status = ?, end_time = CURRENT_TIMESTAMP
    WHERE run_id = ?
    """, ("completed", mega_run_id))
    
    conn.commit()
    conn.close()
    
    print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Mega-result directory created: {mega_dir}")
    print(f"You can now view the mega-result with:")
    print(f"python paper_revision.py --view-mega {mega_run_id}")

if __name__ == "__main__":
    main()