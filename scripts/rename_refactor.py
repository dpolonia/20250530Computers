#!/usr/bin/env python3
"""
Naming convention refactoring tool for the paper revision tool.

This script automatically refactors Python files to fix common naming
convention issues based on the project's naming convention style guide.
"""

import os
import re
import ast
import argparse
import sys
from typing import Dict, List, Set, Tuple, Any, Optional
import libcst as cst
from libcst.metadata import MetadataWrapper


class ParameterRenamer(cst.CSTTransformer):
    """Rename parameters according to naming conventions."""
    
    PARAMETER_MAPPINGS = {
        "file": "file_path",
        "path": "file_path",
        "filepath": "file_path",
        "filename": "file_path",
        "output": "output_path",
        "output_file": "output_path",
        "outpath": "output_path",
        "outfile": "output_path",
        "key": "api_key",
        "apikey": "api_key",
        "api_token": "api_key",
        "model": "model_name",
        "model_id": "model_name",
        "llm": "model_name"
    }
    
    def __init__(self, rename_map: Optional[Dict[str, str]] = None):
        """Initialize the transformer."""
        self.rename_map = rename_map or self.PARAMETER_MAPPINGS
        self.changes = []
    
    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        """Transform parameter nodes."""
        if original_node.name.value in self.rename_map:
            new_name = self.rename_map[original_node.name.value]
            self.changes.append((original_node.name.value, new_name))
            return updated_node.with_changes(name=cst.Name(value=new_name))
        return updated_node


class CamelToSnakeCaseTransformer(cst.CSTTransformer):
    """Transform camelCase names to snake_case."""
    
    def __init__(self):
        """Initialize the transformer."""
        self.changes = []
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Transform function definitions."""
        name = original_node.name.value
        
        # Skip dunder methods
        if name.startswith('__') and name.endswith('__'):
            return updated_node
            
        # Check for camelCase
        if re.search(r'[a-z][A-Z]', name):
            snake_case = self._to_snake_case(name)
            self.changes.append((name, snake_case))
            return updated_node.with_changes(name=cst.Name(value=snake_case))
            
        return updated_node
    
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        """Transform variable names."""
        name = original_node.value
        
        # Skip identifiers that are all uppercase (likely constants)
        if name.isupper():
            return updated_node
            
        # Check for camelCase
        if re.search(r'[a-z][A-Z]', name):
            snake_case = self._to_snake_case(name)
            self.changes.append((name, snake_case))
            return cst.Name(value=snake_case)
            
        return updated_node
    
    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case."""
        # Handle camelCase
        s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        # Handle PascalCase
        s2 = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1_\2', s1)
        # Handle abbreviations (e.g., APIClient -> api_client)
        s3 = re.sub(r'([a-z])([A-Z]+)([A-Z][a-z])', r'\1_\2_\3', s2)
        return s3.lower()


class ClassToPassCaseTransformer(cst.CSTTransformer):
    """Transform snake_case class names to PascalCase."""
    
    def __init__(self):
        """Initialize the transformer."""
        self.changes = []
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Transform class definitions."""
        name = original_node.name.value
        
        # Check for snake_case
        if '_' in name:
            pascal_case = self._to_pascal_case(name)
            self.changes.append((name, pascal_case))
            return updated_node.with_changes(name=cst.Name(value=pascal_case))
            
        return updated_node
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert a name to PascalCase."""
        return ''.join(word.capitalize() for word in name.split('_'))


class AmbiguousFunctionRenamer(cst.CSTTransformer):
    """Rename ambiguous function names."""
    
    RENAME_MAPPINGS = {
        # Core ambiguous names with suggested replacements based on context
        "process": {
            "document": "process_document",
            "text": "process_text",
            "data": "process_data",
            "reference": "process_reference",
            "comment": "process_comment"
        },
        "get": {
            "data": "get_data",
            "text": "get_text",
            "document": "get_document",
            "reference": "get_reference",
            "completion": "get_completion",
            "result": "get_result"
        },
        "handle": {
            "error": "handle_error",
            "response": "handle_response",
            "request": "handle_request",
            "data": "handle_data"
        },
        "check": {
            "valid": "check_validity",
            "format": "check_format",
            "reference": "check_reference",
            "data": "check_data"
        },
        "validate": {
            "data": "validate_data",
            "reference": "validate_reference",
            "format": "validate_format",
            "input": "validate_input"
        }
    }
    
    def __init__(self, rename_map: Optional[Dict[str, Dict[str, str]]] = None):
        """Initialize the transformer."""
        self.rename_map = rename_map or self.RENAME_MAPPINGS
        self.changes = []
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Transform function definitions."""
        name = original_node.name.value
        
        # Skip if already specific enough
        if '_' in name:
            return updated_node
            
        # Check if name is in our ambiguous list
        if name in self.rename_map:
            # Try to determine context from docstring
            docstring = self._get_docstring(original_node)
            if docstring:
                for context, replacement in self.rename_map[name].items():
                    if context.lower() in docstring.lower():
                        self.changes.append((name, replacement))
                        return updated_node.with_changes(name=cst.Name(value=replacement))
            
            # Fallback: use the first parameter's type hint if available
            for param in original_node.params.params:
                if param.annotation:
                    annotation_str = self._get_annotation_str(param.annotation)
                    for context, replacement in self.rename_map[name].items():
                        if context.lower() in annotation_str.lower():
                            self.changes.append((name, replacement))
                            return updated_node.with_changes(name=cst.Name(value=replacement))
                            
            # Final fallback: Use the generic version
            if "generic" in self.rename_map[name]:
                replacement = self.rename_map[name]["generic"]
                self.changes.append((name, replacement))
                return updated_node.with_changes(name=cst.Name(value=replacement))
            
        return updated_node
    
    def _get_docstring(self, node: cst.FunctionDef) -> Optional[str]:
        """Extract docstring from a function node."""
        if node.body.body and isinstance(node.body.body[0], cst.SimpleStatementLine):
            stmt = node.body.body[0]
            if (stmt.body and isinstance(stmt.body[0], cst.Expr) and 
                    isinstance(stmt.body[0].value, cst.SimpleString)):
                return stmt.body[0].value.value
        return None
    
    def _get_annotation_str(self, annotation: cst.Annotation) -> str:
        """Convert an annotation to a string representation."""
        if isinstance(annotation.annotation, cst.Name):
            return annotation.annotation.value
        elif isinstance(annotation.annotation, cst.Attribute):
            return self._get_attribute_str(annotation.annotation)
        elif isinstance(annotation.annotation, cst.Subscript):
            # Handle things like List[str]
            return self._get_subscript_str(annotation.annotation)
        return ""
    
    def _get_attribute_str(self, node: cst.Attribute) -> str:
        """Convert an attribute to a string representation."""
        if isinstance(node.value, cst.Name):
            return f"{node.value.value}.{node.attr.value}"
        elif isinstance(node.value, cst.Attribute):
            return f"{self._get_attribute_str(node.value)}.{node.attr.value}"
        return node.attr.value
    
    def _get_subscript_str(self, node: cst.Subscript) -> str:
        """Convert a subscript to a string representation."""
        if isinstance(node.value, cst.Name):
            return node.value.value
        return ""


def refactor_file(
    file_path: str, 
    fix_parameters: bool = False,
    fix_camel_case: bool = False,
    fix_class_names: bool = False,
    fix_ambiguous: bool = False,
    dry_run: bool = False
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Refactor naming conventions in a file.
    
    Args:
        file_path: Path to the file to refactor
        fix_parameters: Whether to fix parameter names
        fix_camel_case: Whether to fix camelCase names
        fix_class_names: Whether to fix class names
        fix_ambiguous: Whether to fix ambiguous function names
        dry_run: Whether to actually modify the file
        
    Returns:
        Tuple of (changes, new_content)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the code
    module = cst.parse_module(source_code)
    wrapper = MetadataWrapper(module)
    
    changes = []
    
    # Apply transformations
    if fix_parameters:
        transformer = ParameterRenamer()
        modified_tree = wrapper.visit(transformer)
        changes.extend(transformer.changes)
        source_code = modified_tree.code
        
    if fix_camel_case:
        transformer = CamelToSnakeCaseTransformer()
        module = cst.parse_module(source_code)
        wrapper = MetadataWrapper(module)
        modified_tree = wrapper.visit(transformer)
        changes.extend(transformer.changes)
        source_code = modified_tree.code
        
    if fix_class_names:
        transformer = ClassToPassCaseTransformer()
        module = cst.parse_module(source_code)
        wrapper = MetadataWrapper(module)
        modified_tree = wrapper.visit(transformer)
        changes.extend(transformer.changes)
        source_code = modified_tree.code
        
    if fix_ambiguous:
        transformer = AmbiguousFunctionRenamer()
        module = cst.parse_module(source_code)
        wrapper = MetadataWrapper(module)
        modified_tree = wrapper.visit(transformer)
        changes.extend(transformer.changes)
        source_code = modified_tree.code
    
    # Write back the modified code if changes were made and not in dry run mode
    if changes and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(source_code)
    
    return changes, source_code


def refactor_directory(
    directory: str,
    fix_parameters: bool = False,
    fix_camel_case: bool = False,
    fix_class_names: bool = False,
    fix_ambiguous: bool = False,
    dry_run: bool = False
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Refactor naming conventions in all Python files in a directory.
    
    Args:
        directory: Directory to refactor
        fix_parameters: Whether to fix parameter names
        fix_camel_case: Whether to fix camelCase names
        fix_class_names: Whether to fix class names
        fix_ambiguous: Whether to fix ambiguous function names
        dry_run: Whether to actually modify files
        
    Returns:
        Dictionary mapping file paths to lists of changes made
    """
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    changes, _ = refactor_file(
                        file_path, 
                        fix_parameters=fix_parameters,
                        fix_camel_case=fix_camel_case,
                        fix_class_names=fix_class_names,
                        fix_ambiguous=fix_ambiguous,
                        dry_run=dry_run
                    )
                    if changes:
                        results[file_path] = changes
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return results


def main() -> int:
    """Run the rename refactoring tool."""
    parser = argparse.ArgumentParser(description='Refactor naming conventions in Python code')
    parser.add_argument('--directory', '-d', default='src', help='Directory to refactor')
    parser.add_argument('--file', '-f', help='Specific file to refactor')
    parser.add_argument('--parameters', '-p', action='store_true', help='Fix parameter names')
    parser.add_argument('--camel-case', '-c', action='store_true', help='Fix camelCase names')
    parser.add_argument('--class-names', '-n', action='store_true', help='Fix class names')
    parser.add_argument('--ambiguous', '-a', action='store_true', help='Fix ambiguous function names')
    parser.add_argument('--all', action='store_true', help='Fix all naming issues')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without modifying files')
    
    args = parser.parse_args()
    
    # Determine what to fix
    fix_parameters = args.parameters or args.all
    fix_camel_case = args.camel_case or args.all
    fix_class_names = args.class_names or args.all
    fix_ambiguous = args.ambiguous or args.all
    
    # Refactor file or directory
    if args.file:
        changes, _ = refactor_file(
            args.file, 
            fix_parameters=fix_parameters,
            fix_camel_case=fix_camel_case,
            fix_class_names=fix_class_names,
            fix_ambiguous=fix_ambiguous,
            dry_run=args.dry_run
        )
        
        if changes:
            print(f"Changes in {args.file}:")
            for old_name, new_name in changes:
                print(f"  {old_name} -> {new_name}")
        else:
            print(f"No changes needed in {args.file}")
    else:
        results = refactor_directory(
            args.directory, 
            fix_parameters=fix_parameters,
            fix_camel_case=fix_camel_case,
            fix_class_names=fix_class_names,
            fix_ambiguous=fix_ambiguous,
            dry_run=args.dry_run
        )
        
        if results:
            print(f"Changes in {len(results)} files:")
            for file_path, changes in results.items():
                print(f"\n{file_path}:")
                for old_name, new_name in changes:
                    print(f"  {old_name} -> {new_name}")
        else:
            print("No changes needed")
    
    # Print summary
    print("\nSummary:")
    if args.dry_run:
        print("Dry run completed. No files were modified.")
    else:
        print("Refactoring completed.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())