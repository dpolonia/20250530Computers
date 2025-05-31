"""
Query sanitization utilities to prevent SQL injection and other attacks.

This module provides functions to sanitize database queries and other
inputs that might be vulnerable to injection attacks.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class QuerySanitizationError(Exception):
    """Exception raised for query sanitization errors."""
    pass


def sanitize_table_name(name: str) -> str:
    """
    Sanitize a table name to prevent SQL injection.
    
    Args:
        name: Table name to sanitize
        
    Returns:
        Sanitized table name
        
    Raises:
        QuerySanitizationError: If sanitization fails
    """
    if not name:
        raise QuerySanitizationError("Table name cannot be empty")
        
    # Strip whitespace
    name = name.strip()
    
    # Check for valid table name (alphanumeric and underscores only)
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        raise QuerySanitizationError(
            f"Invalid table name: {name}. "
            "Only alphanumeric characters and underscores are allowed."
        )
        
    return name


def sanitize_column_name(name: str) -> str:
    """
    Sanitize a column name to prevent SQL injection.
    
    Args:
        name: Column name to sanitize
        
    Returns:
        Sanitized column name
        
    Raises:
        QuerySanitizationError: If sanitization fails
    """
    if not name:
        raise QuerySanitizationError("Column name cannot be empty")
        
    # Strip whitespace
    name = name.strip()
    
    # Check for valid column name (alphanumeric and underscores only)
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        raise QuerySanitizationError(
            f"Invalid column name: {name}. "
            "Only alphanumeric characters and underscores are allowed."
        )
        
    return name


def create_parameterized_query(
    table_name: str,
    columns: Optional[List[str]] = None,
    where_conditions: Optional[Dict[str, Any]] = None,
    order_by: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Tuple[str, List]:
    """
    Create a parameterized SELECT query to prevent SQL injection.
    
    Args:
        table_name: Name of the table to query
        columns: List of column names to select (default: all columns)
        where_conditions: Dictionary of column-value pairs for WHERE clause
        order_by: List of column names to order by
        limit: Maximum number of rows to return
        
    Returns:
        Tuple of (query string, parameter list)
        
    Raises:
        QuerySanitizationError: If query creation fails
    """
    try:
        # Sanitize table name
        safe_table = sanitize_table_name(table_name)
        
        # Build column list
        column_str = "*"
        if columns:
            safe_columns = [sanitize_column_name(col) for col in columns]
            column_str = ", ".join(safe_columns)
            
        # Start building the query
        query = f"SELECT {column_str} FROM {safe_table}"
        
        # Parameters for prepared statement
        params = []
        
        # Build WHERE clause
        if where_conditions:
            where_clauses = []
            
            for col, value in where_conditions.items():
                # Sanitize column name
                safe_col = sanitize_column_name(col)
                
                # Handle NULL values
                if value is None:
                    where_clauses.append(f"{safe_col} IS NULL")
                else:
                    where_clauses.append(f"{safe_col} = ?")
                    params.append(value)
                    
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
        # Build ORDER BY clause
        if order_by:
            safe_order_columns = []
            
            for col in order_by:
                # Handle DESC suffix
                if col.upper().endswith(" DESC"):
                    col_name = col[:-5].strip()
                    safe_col = sanitize_column_name(col_name)
                    safe_order_columns.append(f"{safe_col} DESC")
                # Handle ASC suffix
                elif col.upper().endswith(" ASC"):
                    col_name = col[:-4].strip()
                    safe_col = sanitize_column_name(col_name)
                    safe_order_columns.append(f"{safe_col} ASC")
                else:
                    safe_col = sanitize_column_name(col)
                    safe_order_columns.append(safe_col)
                    
            if safe_order_columns:
                query += " ORDER BY " + ", ".join(safe_order_columns)
                
        # Add LIMIT clause
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise QuerySanitizationError(
                    f"Invalid LIMIT value: {limit}. Must be a non-negative integer."
                )
                
            query += f" LIMIT {limit}"
            
        return query, params
        
    except Exception as e:
        logger.error(f"Error creating parameterized query: {e}")
        raise QuerySanitizationError(f"Failed to create parameterized query: {e}")


def create_parameterized_insert(
    table_name: str,
    column_values: Dict[str, Any]
) -> Tuple[str, List]:
    """
    Create a parameterized INSERT query to prevent SQL injection.
    
    Args:
        table_name: Name of the table to insert into
        column_values: Dictionary of column-value pairs to insert
        
    Returns:
        Tuple of (query string, parameter list)
        
    Raises:
        QuerySanitizationError: If query creation fails
    """
    if not column_values:
        raise QuerySanitizationError("No column values provided for INSERT")
        
    try:
        # Sanitize table name
        safe_table = sanitize_table_name(table_name)
        
        # Sanitize column names
        safe_columns = [sanitize_column_name(col) for col in column_values.keys()]
        
        # Create placeholders for values
        placeholders = ["?"] * len(safe_columns)
        
        # Build the query
        query = (
            f"INSERT INTO {safe_table} "
            f"({', '.join(safe_columns)}) "
            f"VALUES ({', '.join(placeholders)})"
        )
        
        # Parameters for prepared statement (in the same order as columns)
        params = list(column_values.values())
        
        return query, params
        
    except Exception as e:
        logger.error(f"Error creating parameterized INSERT query: {e}")
        raise QuerySanitizationError(f"Failed to create parameterized INSERT query: {e}")


def create_parameterized_update(
    table_name: str,
    column_values: Dict[str, Any],
    where_conditions: Dict[str, Any]
) -> Tuple[str, List]:
    """
    Create a parameterized UPDATE query to prevent SQL injection.
    
    Args:
        table_name: Name of the table to update
        column_values: Dictionary of column-value pairs to update
        where_conditions: Dictionary of column-value pairs for WHERE clause
        
    Returns:
        Tuple of (query string, parameter list)
        
    Raises:
        QuerySanitizationError: If query creation fails
    """
    if not column_values:
        raise QuerySanitizationError("No column values provided for UPDATE")
        
    if not where_conditions:
        raise QuerySanitizationError(
            "WHERE conditions are required for UPDATE to prevent accidental updates"
        )
        
    try:
        # Sanitize table name
        safe_table = sanitize_table_name(table_name)
        
        # Build SET clause
        set_clauses = []
        params = []
        
        for col, value in column_values.items():
            safe_col = sanitize_column_name(col)
            
            # Handle NULL values
            if value is None:
                set_clauses.append(f"{safe_col} = NULL")
            else:
                set_clauses.append(f"{safe_col} = ?")
                params.append(value)
                
        # Build WHERE clause
        where_clauses = []
        
        for col, value in where_conditions.items():
            safe_col = sanitize_column_name(col)
            
            # Handle NULL values
            if value is None:
                where_clauses.append(f"{safe_col} IS NULL")
            else:
                where_clauses.append(f"{safe_col} = ?")
                params.append(value)
                
        # Build the query
        query = (
            f"UPDATE {safe_table} "
            f"SET {', '.join(set_clauses)} "
            f"WHERE {' AND '.join(where_clauses)}"
        )
        
        return query, params
        
    except Exception as e:
        logger.error(f"Error creating parameterized UPDATE query: {e}")
        raise QuerySanitizationError(f"Failed to create parameterized UPDATE query: {e}")


def create_parameterized_delete(
    table_name: str,
    where_conditions: Dict[str, Any]
) -> Tuple[str, List]:
    """
    Create a parameterized DELETE query to prevent SQL injection.
    
    Args:
        table_name: Name of the table to delete from
        where_conditions: Dictionary of column-value pairs for WHERE clause
        
    Returns:
        Tuple of (query string, parameter list)
        
    Raises:
        QuerySanitizationError: If query creation fails
    """
    if not where_conditions:
        raise QuerySanitizationError(
            "WHERE conditions are required for DELETE to prevent accidental deletion"
        )
        
    try:
        # Sanitize table name
        safe_table = sanitize_table_name(table_name)
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        for col, value in where_conditions.items():
            safe_col = sanitize_column_name(col)
            
            # Handle NULL values
            if value is None:
                where_clauses.append(f"{safe_col} IS NULL")
            else:
                where_clauses.append(f"{safe_col} = ?")
                params.append(value)
                
        # Build the query
        query = (
            f"DELETE FROM {safe_table} "
            f"WHERE {' AND '.join(where_clauses)}"
        )
        
        return query, params
        
    except Exception as e:
        logger.error(f"Error creating parameterized DELETE query: {e}")
        raise QuerySanitizationError(f"Failed to create parameterized DELETE query: {e}")


def execute_query(
    connection,
    query: str,
    params: List = None
) -> List[Dict[str, Any]]:
    """
    Execute a parameterized query safely.
    
    Args:
        connection: Database connection object
        query: Parameterized query string
        params: List of parameter values
        
    Returns:
        List of dictionaries with query results
        
    Raises:
        QuerySanitizationError: If query execution fails
    """
    if params is None:
        params = []
        
    try:
        cursor = connection.cursor()
        cursor.execute(query, params)
        
        # For SELECT queries, fetch results
        if query.strip().upper().startswith("SELECT"):
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        # For other queries, commit changes and return rowcount
        connection.commit()
        return [{"rowcount": cursor.rowcount}]
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        if connection:
            connection.rollback()
        raise QuerySanitizationError(f"Failed to execute query: {e}")
    finally:
        if cursor:
            cursor.close()