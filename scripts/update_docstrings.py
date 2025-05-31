#!/usr/bin/env python3
"""
Docstring updater utility for the paper revision tool.

This script analyzes Python files in the project and updates their module-level
docstrings to follow the project's documentation standards. It can also generate
reports on docstring coverage and quality.
"""

import os
import re
import sys
import argparse
import ast
from typing import Dict, List, Optional, Any, Tuple, Set


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to collect docstring information."""
    
    def __init__(self):
        self.module_docstring = None
        self.classes = {}
        self.functions = {}
        self.current_class = None
    
    def visit_Module(self, node):
        """Visit a module node."""
        self.module_docstring = ast.get_docstring(node)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit a class definition node."""
        old_class = self.current_class
        self.current_class = node.name
        
        self.classes[node.name] = {
            "docstring": ast.get_docstring(node),
            "has_docstring": ast.get_docstring(node) is not None,
            "methods": {}
        }
        
        # Visit class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                docstring = ast.get_docstring(item)
                self.classes[node.name]["methods"][item.name] = {
                    "docstring": docstring,
                    "has_docstring": docstring is not None
                }
        
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Visit a function definition node."""
        # Only process top-level functions, not methods
        if self.current_class is None:
            docstring = ast.get_docstring(node)
            self.functions[node.name] = {
                "docstring": docstring,
                "has_docstring": docstring is not None
            }


def analyze_docstrings(directory: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze docstrings in Python files in the given directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary mapping file paths to docstring analysis results
    """
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            results[file_path] = analyze_file_docstrings(file_path)
    
    return results


def analyze_file_docstrings(file_path: str) -> Dict[str, Any]:
    """
    Analyze docstrings in a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with docstring analysis results
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {
            "module_docstring": None,
            "has_module_docstring": False,
            "classes": {},
            "functions": {},
            "error": "SyntaxError"
        }
    
    visitor = DocstringVisitor()
    visitor.visit(tree)
    
    result = {
        "module_docstring": visitor.module_docstring,
        "has_module_docstring": visitor.module_docstring is not None,
        "classes": visitor.classes,
        "functions": visitor.functions,
        "error": None
    }
    
    return result


def generate_module_docstring(file_path: str) -> str:
    """
    Generate a standard module-level docstring for a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Generated module docstring
    """
    # Extract module name from file path
    module_name = os.path.basename(file_path).replace(".py", "")
    
    # Convert snake_case to words
    module_words = module_name.replace("_", " ")
    
    # Generate a basic docstring
    docstring = f'''"""
{module_words.title()} module for the paper revision tool.

This module provides functionality related to {module_words}.
TODO: Add more detailed description of what this module does.

Classes:
    TODO: List and briefly describe classes in this module

Functions:
    TODO: List and briefly describe functions in this module

Copyright (c) 2025 Paper Revision Tool
"""'''
    
    return docstring


def update_module_docstring(file_path: str, docstring: str) -> bool:
    """
    Update the module-level docstring in a Python file.
    
    Args:
        file_path: Path to the Python file
        docstring: New docstring to add
        
    Returns:
        True if the docstring was updated, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if file has a module docstring
    try:
        tree = ast.parse(content)
        existing_docstring = ast.get_docstring(tree)
    except SyntaxError:
        print(f"Syntax error in {file_path}, skipping")
        return False
    
    if existing_docstring:
        # Replace existing docstring
        docstring_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
        new_content = docstring_pattern.sub(docstring, content, count=1)
    else:
        # Add new docstring at the beginning of the file
        # Check for shebang line
        if content.startswith("#!"):
            shebang_end = content.find("\n") + 1
            new_content = content[:shebang_end] + "\n" + docstring + "\n\n" + content[shebang_end:]
        else:
            new_content = docstring + "\n\n" + content
    
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    
    return False


def generate_docstring_report(analysis_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a report on docstring coverage and quality.
    
    Args:
        analysis_results: Results from analyze_docstrings
        
    Returns:
        Report as a string
    """
    total_files = len(analysis_results)
    files_with_module_docstring = sum(1 for result in analysis_results.values() if result["has_module_docstring"])
    
    total_classes = sum(len(result["classes"]) for result in analysis_results.values())
    classes_with_docstring = sum(sum(1 for cls in result["classes"].values() if cls["has_docstring"]) for result in analysis_results.values())
    
    total_methods = sum(sum(len(cls["methods"]) for cls in result["classes"].values()) for result in analysis_results.values())
    methods_with_docstring = sum(sum(sum(1 for method in cls["methods"].values() if method["has_docstring"]) for cls in result["classes"].values()) for result in analysis_results.values())
    
    total_functions = sum(len(result["functions"]) for result in analysis_results.values())
    functions_with_docstring = sum(sum(1 for func in result["functions"].values() if func["has_docstring"]) for result in analysis_results.values())
    
    # Calculate percentages safely
    class_percent = classes_with_docstring / total_classes * 100 if total_classes > 0 else 0
    method_percent = methods_with_docstring / total_methods * 100 if total_methods > 0 else 0
    function_percent = functions_with_docstring / total_functions * 100 if total_functions > 0 else 0
    
    report = f"""
Docstring Coverage Report
========================

Summary:
- Files analyzed: {total_files}
- Files with module docstring: {files_with_module_docstring} ({files_with_module_docstring / total_files * 100:.1f}% coverage)
- Classes with docstring: {classes_with_docstring} / {total_classes} ({class_percent:.1f}% coverage)
- Methods with docstring: {methods_with_docstring} / {total_methods} ({method_percent:.1f}% coverage)
- Functions with docstring: {functions_with_docstring} / {total_functions} ({function_percent:.1f}% coverage)

Files missing module docstrings:
"""
    
    for file_path, result in sorted(analysis_results.items()):
        if not result["has_module_docstring"]:
            report += f"- {file_path}\n"
    
    return report


def update_all_module_docstrings(directory: str, dry_run: bool = False) -> Dict[str, bool]:
    """
    Update all module docstrings in Python files in the given directory.
    
    Args:
        directory: Directory to process
        dry_run: If True, don't actually update files
        
    Returns:
        Dictionary mapping file paths to update status
    """
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            analysis = analyze_file_docstrings(file_path)
            
            if not analysis["has_module_docstring"]:
                docstring = generate_module_docstring(file_path)
                
                if dry_run:
                    print(f"Would update {file_path} with docstring:\n{docstring}")
                    results[file_path] = True
                else:
                    success = update_module_docstring(file_path, docstring)
                    results[file_path] = success
                    if success:
                        print(f"Updated module docstring in {file_path}")
                    else:
                        print(f"Failed to update module docstring in {file_path}")
    
    return results


def main():
    """Main function to run the docstring updater."""
    parser = argparse.ArgumentParser(description="Update module docstrings in Python files")
    parser.add_argument("--directory", "-d", default="src", help="Directory to process")
    parser.add_argument("--report", "-r", action="store_true", help="Generate docstring coverage report")
    parser.add_argument("--update", "-u", action="store_true", help="Update missing module docstrings")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update files")
    
    args = parser.parse_args()
    
    if args.report:
        analysis_results = analyze_docstrings(args.directory)
        report = generate_docstring_report(analysis_results)
        print(report)
    
    if args.update:
        update_results = update_all_module_docstrings(args.directory, args.dry_run)
        print(f"\nUpdated {sum(1 for success in update_results.values() if success)} files")
    
    if not args.report and not args.update:
        print("No action specified. Use --report or --update.")
        parser.print_help()


if __name__ == "__main__":
    main()