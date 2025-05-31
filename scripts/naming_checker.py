#!/usr/bin/env python3
"""
Naming convention checker for the paper revision tool.

This script analyzes Python files to detect naming convention inconsistencies
and violations based on the project's naming convention style guide.
"""

import os
import re
import ast
import argparse
import sys
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class NamingIssue:
    """Represents a naming convention issue."""
    
    file_path: str
    line_number: int
    issue_type: str
    name: str
    suggestion: Optional[str] = None
    context: Optional[str] = None
    
    def __str__(self) -> str:
        """Format the issue as a string."""
        base = f"{self.file_path}:{self.line_number} - {self.issue_type}: '{self.name}'"
        if self.suggestion:
            base += f" -> Suggestion: '{self.suggestion}'"
        if self.context:
            base += f" (in {self.context})"
        return base


@dataclass
class NamingCheck:
    """Collection of naming issues in a project."""
    
    issues: List[NamingIssue] = field(default_factory=list)
    
    def add_issue(self, issue: NamingIssue) -> None:
        """Add an issue to the collection."""
        self.issues.append(issue)
    
    def sort_issues(self) -> None:
        """Sort issues by file path and line number."""
        self.issues.sort(key=lambda x: (x.file_path, x.line_number))
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of issues by type."""
        summary = {}
        for issue in self.issues:
            if issue.issue_type not in summary:
                summary[issue.issue_type] = 0
            summary[issue.issue_type] += 1
        return summary
    
    def __str__(self) -> str:
        """Format the check results as a string."""
        if not self.issues:
            return "No naming convention issues found."
        
        self.sort_issues()
        
        result = f"Found {len(self.issues)} naming convention issues:\n\n"
        
        # Group by file
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        # Format issues by file
        for file_path, issues in issues_by_file.items():
            result += f"{file_path}:\n"
            for issue in issues:
                result += f"  Line {issue.line_number}: {issue.issue_type}: '{issue.name}'"
                if issue.suggestion:
                    result += f" -> Suggestion: '{issue.suggestion}'"
                if issue.context:
                    result += f" (in {issue.context})"
                result += "\n"
            result += "\n"
        
        # Add summary
        result += "Summary:\n"
        for issue_type, count in self.get_summary().items():
            result += f"  {issue_type}: {count}\n"
        
        return result


class NamingConventionVisitor(ast.NodeVisitor):
    """AST visitor to check naming conventions."""
    
    def __init__(self, file_path: str, check: NamingCheck):
        """Initialize the visitor."""
        self.file_path = file_path
        self.check = check
        self.class_stack = []
        self.function_stack = []
        
        # Common ambiguous names to check
        self.ambiguous_function_names = {
            "process", "handle", "get", "set", "check", "validate", 
            "create", "update", "delete", "find", "extract"
        }
        
        # Parameters that should have consistent names
        self.common_parameters = {
            "file_path": ["file", "path", "filepath", "file_name", "filename"],
            "output_path": ["output", "output_file", "output_name", "output_file_path"],
            "api_key": ["key", "apikey", "api", "token"],
            "model_name": ["model", "model_id", "modelname", "llm", "llm_name"]
        }
        
        # Store function signatures for comparison
        self.function_signatures = {}
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node."""
        # Check class name (should be PascalCase)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.check.add_issue(NamingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="Class name not in PascalCase",
                name=node.name,
                suggestion=self._to_pascal_case(node.name)
            ))
        
        # Check for ambiguous class names
        if len(node.name) <= 3 or node.name in {"Data", "Manager", "Helper", "Util", "Class", "Item", "Object"}:
            self.check.add_issue(NamingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="Ambiguous class name",
                name=node.name
            ))
        
        # Push class onto stack and visit children
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition node."""
        # Determine if this is a method or function
        is_method = bool(self.class_stack)
        context = self.class_stack[-1] if self.class_stack else None
        
        # Skip __special__ methods
        if re.match(r'^__.*__$', node.name):
            self.generic_visit(node)
            return
        
        # Check function/method name (should be snake_case)
        if not re.match(r'^[a-z][a-z0-9_]*$', node.name):
            self.check.add_issue(NamingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="Function/method name not in snake_case",
                name=node.name,
                suggestion=self._to_snake_case(node.name),
                context=f"method in class {context}" if is_method else "function"
            ))
        
        # Check for private methods not using underscore prefix
        if is_method and not node.name.startswith('_') and self._should_be_private(node):
            self.check.add_issue(NamingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="Method should be private",
                name=node.name,
                suggestion=f"_{node.name}",
                context=f"method in class {context}"
            ))
        
        # Check for ambiguous function names without specific context
        if node.name in self.ambiguous_function_names:
            self.check.add_issue(NamingIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                issue_type="Ambiguous function/method name",
                name=node.name,
                context=f"method in class {context}" if is_method else "function"
            ))
        
        # Store function signature for later comparison
        self._store_function_signature(node, is_method, context)
        
        # Check parameter names
        self._check_parameter_names(node, is_method, context)
        
        # Push function onto stack and visit children
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment node."""
        # Check variable names (should be snake_case)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Skip constants (assumed to be uppercase)
                if target.id.isupper():
                    continue
                
                # Check for snake_case
                if not re.match(r'^[a-z][a-z0-9_]*$', target.id):
                    self.check.add_issue(NamingIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="Variable name not in snake_case",
                        name=target.id,
                        suggestion=self._to_snake_case(target.id),
                        context=f"in {'method' if self.function_stack else 'class'} {self.function_stack[-1] if self.function_stack else self.class_stack[-1] if self.class_stack else 'module'}"
                    ))
        
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit a constant node."""
        # Nothing to check for constants
        self.generic_visit(node)
    
    def _should_be_private(self, node: ast.FunctionDef) -> bool:
        """Determine if a method should be private."""
        # Check for helper method patterns
        if any(
            node.name.startswith(prefix) for prefix in 
            ["_", "helper", "internal", "util", "aux", "impl"]
        ):
            return True
        
        # Check docstring for "internal use" or similar
        docstring = ast.get_docstring(node)
        if docstring and any(
            phrase in docstring.lower() for phrase in 
            ["internal", "helper", "not public", "private", "for use by"]
        ):
            return True
        
        return False
    
    def _store_function_signature(self, node: ast.FunctionDef, is_method: bool, context: Optional[str]) -> None:
        """Store function signature for later comparison."""
        signature = {
            "name": node.name,
            "params": [],
            "is_method": is_method,
            "context": context,
            "lineno": node.lineno
        }
        
        # Extract parameter names
        for arg in node.args.args:
            if arg.arg != "self" and arg.arg != "cls":
                signature["params"].append(arg.arg)
        
        # Store signature
        key = context + "." + node.name if context else node.name
        if key not in self.function_signatures:
            self.function_signatures[key] = []
        self.function_signatures[key].append(signature)
    
    def _check_parameter_names(self, node: ast.FunctionDef, is_method: bool, context: Optional[str]) -> None:
        """Check parameter names for consistency and conventions."""
        # Skip self and cls
        params = [arg.arg for arg in node.args.args if arg.arg != "self" and arg.arg != "cls"]
        
        # Check for inconsistent parameter names
        for param in params:
            # Check for camelCase in parameter names
            if re.search(r'[a-z][A-Z]', param):
                self.check.add_issue(NamingIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type="Parameter name not in snake_case",
                    name=param,
                    suggestion=self._to_snake_case(param),
                    context=f"parameter in {'method' if is_method else 'function'} {node.name}"
                ))
            
            # Check for standard parameter names
            for std_name, variants in self.common_parameters.items():
                if param in variants:
                    self.check.add_issue(NamingIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="Non-standard parameter name",
                        name=param,
                        suggestion=std_name,
                        context=f"parameter in {'method' if is_method else 'function'} {node.name}"
                    ))
    
    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case."""
        # Handle camelCase
        s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        # Handle PascalCase
        s2 = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1_\2', s1)
        # Handle abbreviations (e.g., APIClient -> api_client)
        s3 = re.sub(r'([a-z])([A-Z]+)([A-Z][a-z])', r'\1_\2_\3', s2)
        return s3.lower()
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert a name to PascalCase."""
        # Handle snake_case
        words = name.split('_')
        return ''.join(word.capitalize() for word in words)


def check_file_naming_conventions(file_path: str, check: NamingCheck) -> None:
    """
    Check naming conventions in a file.
    
    Args:
        file_path: Path to the file to check
        check: NamingCheck object to store issues
    """
    # Check file name (should be snake_case)
    file_name = os.path.basename(file_path)
    if not re.match(r'^[a-z][a-z0-9_]*\.py$', file_name):
        check.add_issue(NamingIssue(
            file_path=file_path,
            line_number=0,
            issue_type="File name not in snake_case",
            name=file_name,
            suggestion=to_snake_case(file_name)
        ))
    
    # Parse file
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            check.add_issue(NamingIssue(
                file_path=file_path,
                line_number=0,
                issue_type="Syntax error in file",
                name=file_name
            ))
            return
    
    # Visit AST
    visitor = NamingConventionVisitor(file_path, check)
    visitor.visit(tree)
    
    # Check for similar functions with inconsistent parameter names
    check_function_parameter_consistency(visitor.function_signatures, check, file_path)


def check_function_parameter_consistency(
    signatures: Dict[str, List[Dict[str, Any]]], 
    check: NamingCheck,
    file_path: str
) -> None:
    """
    Check for inconsistent parameter names in similar functions.
    
    Args:
        signatures: Function signatures
        check: NamingCheck object to store issues
        file_path: Path to the file being checked
    """
    # Group functions by semantic similarity (stripped of prefixes)
    similar_functions = {}
    
    for key, sigs in signatures.items():
        # Skip single-parameter functions (less likely to have consistency issues)
        if all(len(sig["params"]) <= 1 for sig in sigs):
            continue
        
        # Strip common prefixes to find semantically similar functions
        stripped_name = re.sub(r'^(get|set|create|update|delete|find|check|validate|process|handle|extract)_', '', key.split('.')[-1])
        
        # Skip internal functions
        if stripped_name.startswith('_'):
            continue
        
        if stripped_name not in similar_functions:
            similar_functions[stripped_name] = []
        
        for sig in sigs:
            similar_functions[stripped_name].append(sig)
    
    # Check for inconsistent parameter names in similar functions
    for stripped_name, sigs in similar_functions.items():
        if len(sigs) <= 1:
            continue
        
        # Compare parameter names
        param_sets = []
        for sig in sigs:
            param_sets.append((sig, set(sig["params"])))
        
        # Check for inconsistencies between pairs
        for i in range(len(param_sets)):
            for j in range(i + 1, len(param_sets)):
                sig1, params1 = param_sets[i]
                sig2, params2 = param_sets[j]
                
                # If parameter sets are different sizes, don't compare
                if len(params1) != len(params2):
                    continue
                
                # Check for differences in parameter names
                if params1 != params2:
                    # Get function display names
                    name1 = f"{sig1['context']}.{sig1['name']}" if sig1['context'] else sig1['name']
                    name2 = f"{sig2['context']}.{sig2['name']}" if sig2['context'] else sig2['name']
                    
                    check.add_issue(NamingIssue(
                        file_path=file_path,
                        line_number=sig1["lineno"],
                        issue_type="Inconsistent parameter names",
                        name=name1,
                        context=f"vs {name2} (parameters: {', '.join(sig1['params'])} vs {', '.join(sig2['params'])})"
                    ))


def to_snake_case(name: str) -> str:
    """Convert a name to snake_case."""
    # Remove extension for files
    if '.' in name:
        name_part, ext = name.split('.', 1)
        name = name_part
    else:
        ext = None
    
    # Handle camelCase
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Handle PascalCase
    s2 = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1_\2', s1)
    # Handle abbreviations (e.g., APIClient -> api_client)
    s3 = re.sub(r'([a-z])([A-Z]+)([A-Z][a-z])', r'\1_\2_\3', s2)
    
    result = s3.lower()
    
    # Add extension back for files
    if ext:
        result = f"{result}.{ext}"
    
    return result


def check_directory_naming_conventions(directory: str, check: NamingCheck) -> None:
    """
    Check naming conventions in all Python files in a directory.
    
    Args:
        directory: Directory to check
        check: NamingCheck object to store issues
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                check_file_naming_conventions(file_path, check)


def main() -> int:
    """Run the naming convention checker."""
    parser = argparse.ArgumentParser(description='Check naming conventions in Python code')
    parser.add_argument('--directory', '-d', default='src', help='Directory to check')
    parser.add_argument('--file', '-f', help='Specific file to check')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    # Create check object
    check = NamingCheck()
    
    # Check file or directory
    if args.file:
        check_file_naming_conventions(args.file, check)
    else:
        check_directory_naming_conventions(args.directory, check)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(str(check))
    else:
        print(check)
    
    # Return non-zero exit code if issues found
    return 1 if check.issues else 0


if __name__ == '__main__':
    sys.exit(main())