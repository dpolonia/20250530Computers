"""
Workflow Database Manager

This module provides a SQLite database interface for storing and managing
the workflow data for the paper revision process. It handles storing:
- Preprocessed file information
- Model evaluations and quality metrics
- Revision steps and progress
- Usage statistics and token/cost tracking
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Tuple

class WorkflowDB:
    """SQLite database manager for workflow data."""
    
    def __init__(self, db_path: str = "./.cache/workflow.db"):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create the database tables if they don't exist."""
        # Workflow runs table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT,
            provider TEXT,
            model TEXT,
            operation_mode TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            total_tokens INTEGER,
            total_cost REAL,
            evaluation_tokens INTEGER,
            evaluation_cost REAL,
            settings TEXT
        )
        ''')
        
        # Files table (preprocessed files)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            run_id TEXT,
            original_path TEXT,
            processed_path TEXT,
            file_type TEXT,
            size INTEGER,
            token_estimate INTEGER,
            page_count INTEGER,
            processed_time TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Steps table (workflow steps)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            step_number INTEGER,
            step_name TEXT,
            start_time TEXT,
            end_time TEXT,
            duration REAL,
            status TEXT,
            output_file TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Model evaluations table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            step_id INTEGER,
            timestamp TEXT,
            primary_model TEXT,
            evaluator_model TEXT,
            task_type TEXT,
            quality_score INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            evaluation_tokens INTEGER,
            evaluation_cost REAL,
            details TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (step_id) REFERENCES steps(step_id)
        )
        ''')
        
        # Model completions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS completions (
            completion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            step_id INTEGER,
            timestamp TEXT,
            provider TEXT,
            model TEXT,
            task_type TEXT,
            prompt_hash TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cost REAL,
            cached BOOLEAN,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (step_id) REFERENCES steps(step_id)
        )
        ''')
        
        # Revision changes table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS changes (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            section TEXT,
            reason TEXT,
            old_text TEXT,
            new_text TEXT,
            line_number INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Model scores table for persisting model capability and cost scores
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_scores (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT,
            model_name TEXT,
            capability_score INTEGER,
            cost_efficiency_score INTEGER,
            input_cost REAL,
            output_cost REAL,
            max_tokens INTEGER,
            description TEXT,
            last_updated TEXT,
            UNIQUE(provider, model_name)
        )
        ''')
        
        # Model update schedule table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_updates (
            update_id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_check TEXT,
            next_check TEXT,
            update_interval_days INTEGER DEFAULT 14,
            status TEXT
        )
        ''')
        
        # Reviewer personas table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviewer_personas (
            persona_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            reviewer_id INTEGER,
            persona_index INTEGER,
            archetype TEXT,
            is_primary BOOLEAN,
            is_fine_persona BOOLEAN,
            fine_persona_id TEXT,
            field TEXT,
            focus_areas TEXT,
            tone TEXT,
            depth TEXT,
            expertise TEXT,
            strictness REAL,
            bias TEXT,
            description TEXT,
            background TEXT,
            review_style TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Editor personas table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS editor_personas (
            persona_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            persona_index INTEGER,
            archetype TEXT,
            is_primary BOOLEAN,
            is_fine_persona BOOLEAN,
            fine_persona_id TEXT,
            field TEXT,
            focus_areas TEXT,
            tone TEXT,
            depth TEXT,
            expertise TEXT,
            strictness REAL,
            bias TEXT,
            description TEXT,
            background TEXT,
            editorial_style TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Reviews table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            reviewer_id INTEGER,
            persona_id INTEGER,
            summary TEXT,
            assessment TEXT,
            rating REAL,
            comments TEXT,
            perspective TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (persona_id) REFERENCES reviewer_personas(persona_id)
        )
        ''')
        
        # Editor decisions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS editor_decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            persona_id INTEGER,
            summary_of_reviews TEXT,
            common_concerns TEXT,
            unique_insights TEXT,
            decision TEXT,
            revision_instructions TEXT,
            comments TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (persona_id) REFERENCES editor_personas(persona_id)
        )
        ''')
        
        # Final consolidated decisions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS consolidated_decisions (
            consolidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            decision TEXT,
            summary TEXT,
            concerns TEXT,
            insights TEXT,
            instructions TEXT,
            comments TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        ''')
        
        # Journal information table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS journals (
            journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            issn TEXT UNIQUE,
            e_issn TEXT,
            title TEXT,
            publisher TEXT,
            subject_areas TEXT,
            description TEXT,
            aims_scope TEXT,
            website TEXT,
            open_access BOOLEAN,
            publication_frequency TEXT,
            first_indexed_year INTEGER,
            last_updated TEXT
        )
        ''')
        
        # Journal metrics table (updated yearly)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_id INTEGER,
            year INTEGER,
            sjr REAL,
            snip REAL,
            impact_factor REAL,
            cite_score REAL,
            quartile INTEGER,
            h_index INTEGER,
            total_docs INTEGER,
            total_refs INTEGER,
            total_cites INTEGER,
            percent_cited INTEGER,
            percent_not_cited INTEGER,
            timestamp TEXT,
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
        )
        ''')
        
        # Journal top papers table (most cited/influential papers)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_top_papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_id INTEGER,
            scopus_id TEXT,
            doi TEXT,
            title TEXT,
            authors TEXT,
            publication_year INTEGER,
            volume TEXT,
            issue TEXT,
            pages TEXT,
            citations INTEGER,
            abstract TEXT,
            keywords TEXT,
            ranking_type TEXT,
            ranking_value INTEGER,
            timestamp TEXT,
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
        )
        ''')
        
        # Journal reviewer preferences table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_id INTEGER,
            preference_type TEXT,
            preference_value TEXT,
            importance INTEGER,
            description TEXT,
            timestamp TEXT,
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
        )
        ''')
        
        # Journal submission guidelines
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_guidelines (
            guideline_id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_id INTEGER,
            category TEXT,
            subcategory TEXT,
            content TEXT,
            importance INTEGER,
            timestamp TEXT,
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
        )
        ''')
        
        # Similar papers in the target journal
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS similar_journal_papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            journal_id INTEGER,
            scopus_id TEXT,
            doi TEXT,
            title TEXT,
            authors TEXT,
            publication_year INTEGER,
            volume TEXT,
            issue TEXT,
            pages TEXT,
            citations INTEGER,
            abstract TEXT,
            keywords TEXT,
            similarity_score REAL,
            similarity_reason TEXT,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
        )
        ''')
        
        # Commit the changes
        self.conn.commit()
    
    def create_run(self, run_id: str, provider: str, model: str, operation_mode: str,
                  settings: Dict[str, Any]) -> str:
        """Create a new workflow run.
        
        Args:
            run_id: Unique ID for the run (typically a timestamp)
            provider: Model provider (anthropic, openai, google)
            model: Model name
            operation_mode: Operation mode (training, finetuning, final)
            settings: Dictionary of run settings
            
        Returns:
            The run_id
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO runs (run_id, timestamp, provider, model, operation_mode, 
                         status, start_time, settings)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, now, provider, model, operation_mode, 
             "started", now, json.dumps(settings)))
        
        self.conn.commit()
        return run_id
    
    def complete_run(self, run_id: str, status: str, total_tokens: int, total_cost: float,
                    evaluation_tokens: int = 0, evaluation_cost: float = 0.0):
        """Mark a workflow run as completed.
        
        Args:
            run_id: Unique ID for the run
            status: Final status (completed, failed, etc.)
            total_tokens: Total tokens used
            total_cost: Total cost incurred
            evaluation_tokens: Tokens used for evaluation
            evaluation_cost: Cost incurred for evaluation
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        UPDATE runs 
        SET status = ?, end_time = ?, total_tokens = ?, total_cost = ?,
            evaluation_tokens = ?, evaluation_cost = ?
        WHERE run_id = ?
        ''', (status, now, total_tokens, total_cost, 
             evaluation_tokens, evaluation_cost, run_id))
        
        self.conn.commit()
    
    def get_run(self, run_id: str) -> Dict:
        """Get information about a specific run.
        
        Args:
            run_id: Unique ID for the run
            
        Returns:
            Dictionary with run information
        """
        self.cursor.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,))
        run = self.cursor.fetchone()
        
        if run:
            return dict(run)
        return None
    
    def get_runs(self, limit: int = 10, operation_mode: Optional[str] = None) -> List[Dict]:
        """Get a list of recent runs.
        
        Args:
            limit: Maximum number of runs to return
            operation_mode: Optional filter by operation mode
            
        Returns:
            List of dictionaries with run information
        """
        if operation_mode:
            self.cursor.execute('''
            SELECT * FROM runs 
            WHERE operation_mode = ?
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (operation_mode, limit))
        else:
            self.cursor.execute('''
            SELECT * FROM runs 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
        
        runs = self.cursor.fetchall()
        return [dict(run) for run in runs]
        
    def get_latest_provider_runs(self, operation_mode: str) -> Dict[str, Dict]:
        """Get the latest run for each provider in a specific operation mode.
        
        Args:
            operation_mode: The operation mode to filter runs by
            
        Returns:
            Dictionary mapping provider names to their latest run information
        """
        # Get the latest run for each provider in the specified operation mode
        self.cursor.execute('''
        SELECT r1.* 
        FROM runs r1
        JOIN (
            SELECT provider, MAX(timestamp) as max_timestamp
            FROM runs
            WHERE operation_mode = ? AND status = 'completed'
            GROUP BY provider
        ) r2 ON r1.provider = r2.provider AND r1.timestamp = r2.max_timestamp
        WHERE r1.operation_mode = ? AND r1.status = 'completed'
        ''', (operation_mode, operation_mode))
        
        runs = self.cursor.fetchall()
        
        # Map provider names to run information
        provider_runs = {}
        for run in runs:
            run_dict = dict(run)
            provider_runs[run_dict['provider']] = run_dict
            
        return provider_runs
        
    def merge_run_results(self, source_run_ids: List[str], new_run_id: str, provider: str, model: str, 
                       operation_mode: str, settings: Dict[str, Any]) -> str:
        """Create a new merged run from multiple source runs.
        
        Args:
            source_run_ids: List of run IDs to merge from
            new_run_id: ID for the new merged run
            provider: Provider for the merged run
            model: Model for the merged run
            operation_mode: Operation mode for the merged run
            settings: Settings for the merged run
            
        Returns:
            The new run ID
        """
        # Create the new run
        self.create_run(new_run_id, provider, model, operation_mode, settings)
        
        # Copy files from source runs
        for source_run_id in source_run_ids:
            # Get files from the source run
            self.cursor.execute('SELECT * FROM files WHERE run_id = ?', (source_run_id,))
            files = self.cursor.fetchall()
            
            # Copy each file to the new run
            for file in files:
                file_dict = dict(file)
                # Generate a new file ID for the merged run
                new_file_id = file_dict['file_id'] + f"_merged_{new_run_id}"
                
                self.add_file(
                    new_run_id,
                    new_file_id,
                    file_dict['original_path'],
                    file_dict['processed_path'],
                    file_dict['file_type'],
                    file_dict['size'],
                    file_dict['token_estimate'],
                    file_dict['page_count']
                )
            
            # Copy changes from the source run
            self.cursor.execute('SELECT * FROM changes WHERE run_id = ?', (source_run_id,))
            changes = self.cursor.fetchall()
            
            # Copy each change to the new run
            for change in changes:
                change_dict = dict(change)
                self.add_change(
                    new_run_id,
                    change_dict['section'],
                    f"Merged from {source_run_id}: {change_dict['reason']}",
                    change_dict['old_text'],
                    change_dict['new_text'],
                    change_dict['line_number']
                )
        
        return new_run_id
    
    def start_step(self, run_id: str, step_number: int, step_name: str) -> int:
        """Start a new workflow step.
        
        Args:
            run_id: Unique ID for the run
            step_number: The step number in the workflow
            step_name: Name/description of the step
            
        Returns:
            The step_id
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO steps (run_id, step_number, step_name, start_time, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (run_id, step_number, step_name, now, "running"))
        
        step_id = self.cursor.lastrowid
        self.conn.commit()
        return step_id
    
    def complete_step(self, step_id: int, status: str, output_file: Optional[str] = None):
        """Complete a workflow step.
        
        Args:
            step_id: ID of the step to complete
            status: Final status (completed, failed, etc.)
            output_file: Path to the output file (if any)
        """
        now = datetime.datetime.now().isoformat()
        
        # Get the start time to calculate duration
        self.cursor.execute('SELECT start_time FROM steps WHERE step_id = ?', (step_id,))
        step = self.cursor.fetchone()
        
        if step:
            start_time = datetime.datetime.fromisoformat(step['start_time'])
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.cursor.execute('''
            UPDATE steps 
            SET status = ?, end_time = ?, duration = ?, output_file = ?
            WHERE step_id = ?
            ''', (status, now, duration, output_file, step_id))
            
            self.conn.commit()
    
    def add_file(self, run_id: str, file_id: str, original_path: str, 
                processed_path: str, file_type: str, size: int,
                token_estimate: int, page_count: int = 0):
        """Add a preprocessed file.
        
        Args:
            run_id: Unique ID for the run
            file_id: Unique ID for the file
            original_path: Path to the original file
            processed_path: Path to the processed file
            file_type: Type of file (pdf, docx, etc.)
            size: Size of the file in bytes
            token_estimate: Estimated token count
            page_count: Number of pages (for PDFs and documents)
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO files (file_id, run_id, original_path, processed_path, 
                          file_type, size, token_estimate, page_count, processed_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file_id, run_id, original_path, processed_path, 
             file_type, size, token_estimate, page_count, now))
        
        self.conn.commit()
    
    def get_files(self, run_id: str) -> List[Dict]:
        """Get all files for a specific run.
        
        Args:
            run_id: Unique ID for the run
            
        Returns:
            List of dictionaries with file information
        """
        self.cursor.execute('SELECT * FROM files WHERE run_id = ?', (run_id,))
        files = self.cursor.fetchall()
        return [dict(file) for file in files]
    
    def add_evaluation(self, run_id: str, step_id: int, primary_model: str,
                      evaluator_model: str, task_type: str, quality_score: int,
                      prompt_tokens: int, completion_tokens: int,
                      evaluation_tokens: int, evaluation_cost: float,
                      details: Dict[str, Any]):
        """Add a model evaluation.
        
        Args:
            run_id: Unique ID for the run
            step_id: ID of the step
            primary_model: Primary model being evaluated
            evaluator_model: Model doing the evaluation
            task_type: Type of task being evaluated
            quality_score: Quality score (1-5)
            prompt_tokens: Tokens in the prompt
            completion_tokens: Tokens in the completion
            evaluation_tokens: Tokens used for evaluation
            evaluation_cost: Cost of the evaluation
            details: Dictionary with detailed evaluation info
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO evaluations (run_id, step_id, timestamp, primary_model,
                               evaluator_model, task_type, quality_score,
                               prompt_tokens, completion_tokens, evaluation_tokens,
                               evaluation_cost, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, step_id, now, primary_model, evaluator_model, task_type,
             quality_score, prompt_tokens, completion_tokens, evaluation_tokens,
             evaluation_cost, json.dumps(details)))
        
        self.conn.commit()
    
    def add_completion(self, run_id: str, step_id: int, provider: str, model: str,
                      task_type: str, prompt_hash: str, prompt_tokens: int,
                      completion_tokens: int, cost: float, cached: bool = False):
        """Add a model completion.
        
        Args:
            run_id: Unique ID for the run
            step_id: ID of the step
            provider: Model provider
            model: Model name
            task_type: Type of task
            prompt_hash: Hash of the prompt
            prompt_tokens: Tokens in the prompt
            completion_tokens: Tokens in the completion
            cost: Cost of the completion
            cached: Whether the completion was cached
        """
        now = datetime.datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO completions (run_id, step_id, timestamp, provider, model,
                               task_type, prompt_hash, prompt_tokens,
                               completion_tokens, cost, cached)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, step_id, now, provider, model, task_type, prompt_hash,
             prompt_tokens, completion_tokens, cost, cached))
        
        self.conn.commit()
    
    def add_change(self, run_id: str, section: str, reason: str, 
                  old_text: str, new_text: str, line_number: Optional[int] = None):
        """Add a revision change.
        
        Args:
            run_id: Unique ID for the run
            section: Section of the paper
            reason: Reason for the change
            old_text: Original text
            new_text: New text
            line_number: Line number in the original document
        """
        self.cursor.execute('''
        INSERT INTO changes (run_id, section, reason, old_text, new_text, line_number)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (run_id, section, reason, old_text, new_text, line_number))
        
        self.conn.commit()
    
    def get_changes(self, run_id: str) -> List[Dict]:
        """Get all changes for a specific run.
        
        Args:
            run_id: Unique ID for the run
            
        Returns:
            List of dictionaries with change information
        """
        self.cursor.execute('SELECT * FROM changes WHERE run_id = ?', (run_id,))
        changes = self.cursor.fetchall()
        return [dict(change) for change in changes]
    
    def get_stats(self, run_id: str) -> Dict[str, Any]:
        """Get statistics for a specific run.
        
        Args:
            run_id: Unique ID for the run
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Get run information
        self.cursor.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,))
        run = self.cursor.fetchone()
        if run:
            stats['run'] = dict(run)
        
        # Get file counts
        self.cursor.execute('SELECT COUNT(*) as file_count FROM files WHERE run_id = ?', (run_id,))
        file_count = self.cursor.fetchone()
        stats['file_count'] = file_count['file_count'] if file_count else 0
        
        # Get step counts
        self.cursor.execute('''
        SELECT 
            COUNT(*) as step_count,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_steps,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_steps,
            SUM(duration) as total_duration
        FROM steps 
        WHERE run_id = ?
        ''', (run_id,))
        step_stats = self.cursor.fetchone()
        if step_stats:
            stats['step_count'] = step_stats['step_count']
            stats['completed_steps'] = step_stats['completed_steps']
            stats['failed_steps'] = step_stats['failed_steps']
            stats['total_duration'] = step_stats['total_duration']
        
        # Get completion stats
        self.cursor.execute('''
        SELECT 
            COUNT(*) as completion_count,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(cost) as total_cost,
            SUM(CASE WHEN cached THEN 1 ELSE 0 END) as cached_count
        FROM completions 
        WHERE run_id = ?
        ''', (run_id,))
        completion_stats = self.cursor.fetchone()
        if completion_stats:
            stats['completion_count'] = completion_stats['completion_count']
            stats['total_prompt_tokens'] = completion_stats['total_prompt_tokens']
            stats['total_completion_tokens'] = completion_stats['total_completion_tokens']
            stats['total_completion_cost'] = completion_stats['total_cost']
            stats['cached_count'] = completion_stats['cached_count']
        
        # Get evaluation stats
        self.cursor.execute('''
        SELECT 
            COUNT(*) as evaluation_count,
            AVG(quality_score) as avg_quality_score,
            SUM(evaluation_tokens) as total_evaluation_tokens,
            SUM(evaluation_cost) as total_evaluation_cost
        FROM evaluations 
        WHERE run_id = ?
        ''', (run_id,))
        evaluation_stats = self.cursor.fetchone()
        if evaluation_stats:
            stats['evaluation_count'] = evaluation_stats['evaluation_count']
            stats['avg_quality_score'] = evaluation_stats['avg_quality_score']
            stats['total_evaluation_tokens'] = evaluation_stats['total_evaluation_tokens']
            stats['total_evaluation_cost'] = evaluation_stats['total_evaluation_cost']
        
        # Get change stats
        self.cursor.execute('SELECT COUNT(*) as change_count FROM changes WHERE run_id = ?', (run_id,))
        change_count = self.cursor.fetchone()
        stats['change_count'] = change_count['change_count'] if change_count else 0
        
        return stats
    
    def store_model_score(self, provider: str, model_name: str, capability_score: int,
                        cost_efficiency_score: int, input_cost: float, output_cost: float,
                        max_tokens: int, description: str):
        """Store a model score in the database.
        
        Args:
            provider: The provider name ('anthropic', 'openai', 'google')
            model_name: The model name
            capability_score: Capability score (0-100)
            cost_efficiency_score: Cost efficiency score (0-100)
            input_cost: Input cost per 1K tokens
            output_cost: Output cost per 1K tokens
            max_tokens: Maximum tokens supported by the model
            description: Model description
        """
        now = datetime.datetime.now().isoformat()
        
        # Check if the model already exists
        self.cursor.execute(
            "SELECT score_id FROM model_scores WHERE provider = ? AND model_name = ?",
            (provider, model_name)
        )
        existing_score = self.cursor.fetchone()
        
        if existing_score:
            # Update existing record
            self.cursor.execute('''
            UPDATE model_scores
            SET capability_score = ?,
                cost_efficiency_score = ?,
                input_cost = ?,
                output_cost = ?,
                max_tokens = ?,
                description = ?,
                last_updated = ?
            WHERE provider = ? AND model_name = ?
            ''', (capability_score, cost_efficiency_score, input_cost, output_cost,
                  max_tokens, description, now, provider, model_name))
        else:
            # Insert new record
            self.cursor.execute('''
            INSERT INTO model_scores (
                provider, model_name, capability_score, cost_efficiency_score,
                input_cost, output_cost, max_tokens, description, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (provider, model_name, capability_score, cost_efficiency_score,
                  input_cost, output_cost, max_tokens, description, now))
        
        self.conn.commit()
    
    def get_model_scores_from_db(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get all model scores from the database.
        
        Returns:
            Dictionary mapping provider/model to score details
        """
        scores = {}
        
        self.cursor.execute('''
        SELECT provider, model_name, capability_score, cost_efficiency_score,
               input_cost, output_cost, max_tokens, description, last_updated
        FROM model_scores
        ''')
        
        rows = self.cursor.fetchall()
        
        for row in rows:
            provider = row[0]
            model_name = row[1]
            
            if provider not in scores:
                scores[provider] = {}
                
            scores[provider][model_name] = {
                "capability": row[2],
                "cost_efficiency": row[3],
                "input_cost": row[4],
                "output_cost": row[5],
                "max_tokens": row[6],
                "description": row[7],
                "last_updated": row[8]
            }
        
        return scores
    
    def get_model_update_schedule(self) -> Dict[str, Any]:
        """Get the model update schedule.
        
        Returns:
            Dictionary with schedule information
        """
        self.cursor.execute("SELECT * FROM model_updates ORDER BY update_id DESC LIMIT 1")
        row = self.cursor.fetchone()
        
        if row:
            return dict(row)
        else:
            # Initialize the schedule if it doesn't exist
            now = datetime.datetime.now()
            next_check = now + datetime.timedelta(days=14)
            
            self.cursor.execute('''
            INSERT INTO model_updates (last_check, next_check, update_interval_days, status)
            VALUES (?, ?, ?, ?)
            ''', (now.isoformat(), next_check.isoformat(), 14, "initialized"))
            
            self.conn.commit()
            
            return {
                "update_id": 1,
                "last_check": now.isoformat(),
                "next_check": next_check.isoformat(),
                "update_interval_days": 14,
                "status": "initialized"
            }
    
    def update_model_check_schedule(self, status: str = "completed"):
        """Update the model check schedule after a check.
        
        Args:
            status: Status of the check
        """
        now = datetime.datetime.now()
        
        # Get current schedule
        self.cursor.execute("SELECT update_interval_days FROM model_updates ORDER BY update_id DESC LIMIT 1")
        row = self.cursor.fetchone()
        
        if row:
            interval_days = row[0]
        else:
            interval_days = 14
        
        # Calculate next check
        next_check = now + datetime.timedelta(days=interval_days)
        
        # Insert new schedule
        self.cursor.execute('''
        INSERT INTO model_updates (last_check, next_check, update_interval_days, status)
        VALUES (?, ?, ?, ?)
        ''', (now.isoformat(), next_check.isoformat(), interval_days, status))
        
        self.conn.commit()
    
    def is_model_update_due(self) -> bool:
        """Check if a model update is due.
        
        Returns:
            True if an update is due, False otherwise
        """
        schedule = self.get_model_update_schedule()
        
        if not schedule:
            return True
        
        now = datetime.datetime.now()
        next_check = datetime.datetime.fromisoformat(schedule["next_check"])
        
        return now >= next_check
    
    # Methods for storing reviewer personas and reviews
    
    def store_reviewer_persona(self, run_id: str, reviewer_id: int, persona_data: Dict[str, Any]) -> int:
        """Store a reviewer persona in the database.
        
        Args:
            run_id: The run ID
            reviewer_id: The reviewer ID
            persona_data: Dictionary with persona details
            
        Returns:
            The persona ID
        """
        now = datetime.datetime.now().isoformat()
        
        # Extract fine persona ID if applicable
        fine_persona_id = None
        if persona_data.get("is_fine_persona", False) and persona_data.get("archetype"):
            if persona_data["archetype"].startswith("fine_persona_"):
                fine_persona_id = persona_data["archetype"].replace("fine_persona_", "")
        
        # Convert list fields to JSON strings
        focus_areas = json.dumps(persona_data.get("focus_areas", []))
        expertise = json.dumps(persona_data.get("expertise", []))
        
        self.cursor.execute('''
        INSERT INTO reviewer_personas (
            run_id, reviewer_id, persona_index, archetype, is_primary, is_fine_persona,
            fine_persona_id, field, focus_areas, tone, depth, expertise, strictness,
            bias, description, background, review_style, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, reviewer_id, persona_data.get("persona_index", 1),
            persona_data.get("archetype", ""), persona_data.get("is_primary", False),
            persona_data.get("is_fine_persona", False), fine_persona_id,
            persona_data.get("field", ""), focus_areas, persona_data.get("tone", ""),
            persona_data.get("depth", ""), expertise, persona_data.get("strictness", 0.5),
            persona_data.get("bias", ""), persona_data.get("fine_persona_description", ""),
            persona_data.get("background", ""), persona_data.get("review_style", ""),
            now
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def store_review(self, run_id: str, reviewer_id: int, persona_id: int, review_data: Dict[str, Any]) -> int:
        """Store a review in the database.
        
        Args:
            run_id: The run ID
            reviewer_id: The reviewer ID
            persona_id: The persona ID
            review_data: Dictionary with review details
            
        Returns:
            The review ID
        """
        now = datetime.datetime.now().isoformat()
        
        # Convert list fields to JSON strings
        comments = json.dumps({
            "major_comments": review_data.get("major_comments", []),
            "minor_comments": review_data.get("minor_comments", [])
        })
        
        self.cursor.execute('''
        INSERT INTO reviews (
            run_id, reviewer_id, persona_id, summary, assessment, rating,
            comments, perspective, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, reviewer_id, persona_id, review_data.get("summary", ""),
            review_data.get("overall_assessment", ""), review_data.get("rating", 0),
            comments, review_data.get("reviewer_perspective", ""), now
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    # Methods for storing editor personas and decisions
    
    def store_editor_persona(self, run_id: str, persona_data: Dict[str, Any]) -> int:
        """Store an editor persona in the database.
        
        Args:
            run_id: The run ID
            persona_data: Dictionary with persona details
            
        Returns:
            The persona ID
        """
        now = datetime.datetime.now().isoformat()
        
        # Extract fine persona ID if applicable
        fine_persona_id = None
        if persona_data.get("is_fine_persona", False) and persona_data.get("archetype"):
            if persona_data["archetype"].startswith("fine_persona_editor_"):
                fine_persona_id = persona_data["archetype"].replace("fine_persona_editor_", "")
        
        # Convert list fields to JSON strings
        focus_areas = json.dumps(persona_data.get("focus_areas", []))
        expertise = json.dumps(persona_data.get("expertise", []))
        
        self.cursor.execute('''
        INSERT INTO editor_personas (
            run_id, persona_index, archetype, is_primary, is_fine_persona,
            fine_persona_id, field, focus_areas, tone, depth, expertise, 
            strictness, bias, description, background, editorial_style, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, persona_data.get("persona_index", 1),
            persona_data.get("archetype", ""), persona_data.get("is_primary", False),
            persona_data.get("is_fine_persona", False), fine_persona_id,
            persona_data.get("field", ""), focus_areas, persona_data.get("tone", ""),
            persona_data.get("depth", ""), expertise, persona_data.get("strictness", 0.5),
            persona_data.get("bias", ""), persona_data.get("fine_persona_description", ""),
            persona_data.get("background", ""), persona_data.get("editorial_style", ""),
            now
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def store_editor_decision(self, run_id: str, persona_id: int, decision_data: Dict[str, Any]) -> int:
        """Store an editor decision in the database.
        
        Args:
            run_id: The run ID
            persona_id: The editor persona ID
            decision_data: Dictionary with decision details
            
        Returns:
            The decision ID
        """
        now = datetime.datetime.now().isoformat()
        
        # Convert list fields to JSON strings
        common_concerns = json.dumps(decision_data.get("common_concerns", []))
        unique_insights = json.dumps(decision_data.get("unique_insights", []))
        revision_instructions = json.dumps(decision_data.get("revision_instructions", []))
        
        self.cursor.execute('''
        INSERT INTO editor_decisions (
            run_id, persona_id, summary_of_reviews, common_concerns, 
            unique_insights, decision, revision_instructions, comments, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, persona_id, decision_data.get("summary_of_reviews", ""),
            common_concerns, unique_insights, decision_data.get("decision", ""),
            revision_instructions, decision_data.get("editor_comments", ""), now
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def store_consolidated_decision(self, run_id: str, decision_data: Dict[str, Any]) -> int:
        """Store a consolidated editor decision in the database.
        
        Args:
            run_id: The run ID
            decision_data: Dictionary with consolidated decision details
            
        Returns:
            The consolidation ID
        """
        now = datetime.datetime.now().isoformat()
        
        # Convert list fields to JSON strings
        concerns = json.dumps(decision_data.get("common_concerns", []))
        insights = json.dumps(decision_data.get("unique_insights", []))
        instructions = json.dumps(decision_data.get("revision_instructions", []))
        
        self.cursor.execute('''
        INSERT INTO consolidated_decisions (
            run_id, decision, summary, concerns, insights, 
            instructions, comments, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, decision_data.get("decision", ""),
            decision_data.get("summary_of_reviews", ""), concerns,
            insights, instructions, decision_data.get("editor_comments", ""), now
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_review_process_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a summary of the entire review process for a run.
        
        Args:
            run_id: The run ID
            
        Returns:
            Dictionary with review process summary
        """
        summary = {
            "reviewer_count": 0,
            "total_reviewer_personas": 0,
            "editor_count": 0,
            "fine_personas_used": 0,
            "review_count": 0,
            "decision": "",
            "journal_info": {},
            "process_description": ""
        }
        
        # Get journal information if available
        self.cursor.execute("""
        SELECT j.* 
        FROM journals j
        JOIN similar_journal_papers sjp ON j.journal_id = sjp.journal_id
        WHERE sjp.run_id = ?
        LIMIT 1
        """, (run_id,))
        
        journal_row = self.cursor.fetchone()
        if journal_row:
            journal_info = dict(journal_row)
            
            # Get journal metrics
            self.cursor.execute("""
            SELECT * FROM journal_metrics
            WHERE journal_id = ?
            ORDER BY year DESC
            LIMIT 1
            """, (journal_info['journal_id'],))
            
            metrics_row = self.cursor.fetchone()
            if metrics_row:
                journal_info['metrics'] = dict(metrics_row)
                
            summary["journal_info"] = journal_info
        
        # Count reviewer personas
        self.cursor.execute(
            "SELECT COUNT(DISTINCT reviewer_id) FROM reviewer_personas WHERE run_id = ?", 
            (run_id,)
        )
        result = self.cursor.fetchone()
        if result:
            summary["reviewer_count"] = result[0]
            
        self.cursor.execute(
            "SELECT COUNT(*) FROM reviewer_personas WHERE run_id = ?", 
            (run_id,)
        )
        result = self.cursor.fetchone()
        if result:
            summary["total_reviewer_personas"] = result[0]
        
        # Count editor personas
        self.cursor.execute(
            "SELECT COUNT(*) FROM editor_personas WHERE run_id = ?", 
            (run_id,)
        )
        result = self.cursor.fetchone()
        if result:
            summary["editor_count"] = result[0]
            
        # Count fine personas
        self.cursor.execute(
            "SELECT COUNT(*) FROM reviewer_personas WHERE run_id = ? AND is_fine_persona = 1", 
            (run_id,)
        )
        reviewer_fine_count = self.cursor.fetchone()[0] if self.cursor.fetchone() else 0
        
        self.cursor.execute(
            "SELECT COUNT(*) FROM editor_personas WHERE run_id = ? AND is_fine_persona = 1", 
            (run_id,)
        )
        editor_fine_count = self.cursor.fetchone()[0] if self.cursor.fetchone() else 0
        
        summary["fine_personas_used"] = reviewer_fine_count + editor_fine_count
        
        # Count reviews
        self.cursor.execute(
            "SELECT COUNT(*) FROM reviews WHERE run_id = ?", 
            (run_id,)
        )
        result = self.cursor.fetchone()
        if result:
            summary["review_count"] = result[0]
            
        # Get final decision
        self.cursor.execute(
            "SELECT decision FROM consolidated_decisions WHERE run_id = ? ORDER BY timestamp DESC LIMIT 1", 
            (run_id,)
        )
        result = self.cursor.fetchone()
        if result:
            summary["decision"] = result[0]
            
        # Generate process description
        process_description = f"""
        This paper underwent a comprehensive multi-persona review process involving {summary['reviewer_count']} reviewers, 
        each using {summary['total_reviewer_personas'] // max(1, summary['reviewer_count'])} distinct perspectives to evaluate the manuscript. 
        The review board consisted of {summary['editor_count']} editors with diverse academic backgrounds and editorial approaches.
        """
        
        # Add journal information if available
        if "journal_info" in summary and summary["journal_info"]:
            journal_info = summary["journal_info"]
            process_description += f"""
        
        TARGET JOURNAL: {journal_info.get('title', 'Unknown Journal')}
        """
            
            if "metrics" in journal_info:
                metrics = journal_info["metrics"]
                process_description += f"""
        • Impact Factor: {metrics.get('impact_factor', 'N/A')}
        • SJR: {metrics.get('sjr', 'N/A')}
        • SNIP: {metrics.get('snip', 'N/A')}
        • Subject areas: {journal_info.get('subject_areas', 'N/A')}
        """
            
            process_description += f"""
        The revision process was specifically tailored to the scope and requirements of {journal_info.get('title', 'the journal')},
        including analysis of the journal's highly-cited papers and editorial preferences.
        """
        
        process_description += f"""
        
        The multi-persona system leverages both predefined academic archetypes and personas from the FinePersonas dataset
        (https://huggingface.co/datasets/argilla/FinePersonas-v0.1) to ensure a thorough, multi-faceted evaluation.
        In total, {summary['review_count']} individual reviews were generated and consolidated into a final consensus.
        
        Bibliographic data was enhanced using the Scopus API (https://dev.elsevier.com) to ensure accurate citations,
        reference validation, and alignment with the journal's citation patterns.
        
        Final decision: {summary['decision']}
        """
        
        summary["process_description"] = process_description
        
        return summary
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    # Journal-related methods
    
    def store_journal(self, journal_data: Dict[str, Any]) -> int:
        """Store journal information in the database.
        
        Args:
            journal_data: Dictionary with journal information
            
        Returns:
            Journal ID
        """
        # Check if journal already exists
        self.cursor.execute("""
        SELECT journal_id FROM journals
        WHERE issn = ? OR (title = ? AND publisher = ?)
        """, (journal_data.get('issn', ''), journal_data.get('title', ''), 
              journal_data.get('publisher', '')))
        
        result = self.cursor.fetchone()
        if result:
            journal_id = result['journal_id']
            
            # Update existing journal record
            self.cursor.execute("""
            UPDATE journals
            SET e_issn = ?,
                title = ?,
                publisher = ?,
                subject_areas = ?,
                description = ?,
                aims_scope = ?,
                website = ?,
                open_access = ?,
                publication_frequency = ?,
                first_indexed_year = ?,
                last_updated = ?
            WHERE journal_id = ?
            """, (
                journal_data.get('e_issn', ''),
                journal_data.get('title', ''),
                journal_data.get('publisher', ''),
                journal_data.get('subject_areas', ''),
                journal_data.get('description', ''),
                journal_data.get('aims_scope', ''),
                journal_data.get('website', ''),
                journal_data.get('open_access', False),
                journal_data.get('publication_frequency', ''),
                journal_data.get('first_indexed_year', 0),
                datetime.datetime.now().isoformat(),
                journal_id
            ))
        else:
            # Insert new journal record
            self.cursor.execute("""
            INSERT INTO journals (
                issn, e_issn, title, publisher, subject_areas,
                description, aims_scope, website, open_access,
                publication_frequency, first_indexed_year, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                journal_data.get('issn', ''),
                journal_data.get('e_issn', ''),
                journal_data.get('title', ''),
                journal_data.get('publisher', ''),
                journal_data.get('subject_areas', ''),
                journal_data.get('description', ''),
                journal_data.get('aims_scope', ''),
                journal_data.get('website', ''),
                journal_data.get('open_access', False),
                journal_data.get('publication_frequency', ''),
                journal_data.get('first_indexed_year', 0),
                datetime.datetime.now().isoformat()
            ))
            
            journal_id = self.cursor.lastrowid
        
        self.conn.commit()
        return journal_id
    
    def store_journal_metrics(self, journal_id: int, metrics_data: Dict[str, Any]) -> int:
        """Store journal metrics in the database.
        
        Args:
            journal_id: Journal ID
            metrics_data: Dictionary with metrics data
            
        Returns:
            Metric ID
        """
        year = metrics_data.get('year', datetime.datetime.now().year)
        
        # Check if metrics for this year already exist
        self.cursor.execute("""
        SELECT metric_id FROM journal_metrics
        WHERE journal_id = ? AND year = ?
        """, (journal_id, year))
        
        result = self.cursor.fetchone()
        if result:
            metric_id = result['metric_id']
            
            # Update existing metrics
            self.cursor.execute("""
            UPDATE journal_metrics
            SET sjr = ?,
                snip = ?,
                impact_factor = ?,
                cite_score = ?,
                quartile = ?,
                h_index = ?,
                total_docs = ?,
                total_refs = ?,
                total_cites = ?,
                percent_cited = ?,
                percent_not_cited = ?,
                timestamp = ?
            WHERE metric_id = ?
            """, (
                metrics_data.get('sjr', 0.0),
                metrics_data.get('snip', 0.0),
                metrics_data.get('impact_factor', 0.0),
                metrics_data.get('cite_score', 0.0),
                metrics_data.get('quartile', 0),
                metrics_data.get('h_index', 0),
                metrics_data.get('total_docs', 0),
                metrics_data.get('total_refs', 0),
                metrics_data.get('total_cites', 0),
                metrics_data.get('percent_cited', 0),
                metrics_data.get('percent_not_cited', 0),
                datetime.datetime.now().isoformat(),
                metric_id
            ))
        else:
            # Insert new metrics
            self.cursor.execute("""
            INSERT INTO journal_metrics (
                journal_id, year, sjr, snip, impact_factor,
                cite_score, quartile, h_index, total_docs,
                total_refs, total_cites, percent_cited,
                percent_not_cited, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                journal_id,
                year,
                metrics_data.get('sjr', 0.0),
                metrics_data.get('snip', 0.0),
                metrics_data.get('impact_factor', 0.0),
                metrics_data.get('cite_score', 0.0),
                metrics_data.get('quartile', 0),
                metrics_data.get('h_index', 0),
                metrics_data.get('total_docs', 0),
                metrics_data.get('total_refs', 0),
                metrics_data.get('total_cites', 0),
                metrics_data.get('percent_cited', 0),
                metrics_data.get('percent_not_cited', 0),
                datetime.datetime.now().isoformat()
            ))
            
            metric_id = self.cursor.lastrowid
        
        self.conn.commit()
        return metric_id
    
    def store_journal_top_paper(self, journal_id: int, paper_data: Dict[str, Any]) -> int:
        """Store a top paper for a journal.
        
        Args:
            journal_id: Journal ID
            paper_data: Dictionary with paper information
            
        Returns:
            Paper ID
        """
        # Check if paper already exists
        self.cursor.execute("""
        SELECT paper_id FROM journal_top_papers
        WHERE journal_id = ? AND (scopus_id = ? OR doi = ?)
        """, (journal_id, paper_data.get('scopus_id', ''), paper_data.get('doi', '')))
        
        result = self.cursor.fetchone()
        if result:
            paper_id = result['paper_id']
            
            # Update existing paper
            self.cursor.execute("""
            UPDATE journal_top_papers
            SET title = ?,
                authors = ?,
                publication_year = ?,
                volume = ?,
                issue = ?,
                pages = ?,
                citations = ?,
                abstract = ?,
                keywords = ?,
                ranking_type = ?,
                ranking_value = ?,
                timestamp = ?
            WHERE paper_id = ?
            """, (
                paper_data.get('title', ''),
                paper_data.get('authors', ''),
                paper_data.get('publication_year', 0),
                paper_data.get('volume', ''),
                paper_data.get('issue', ''),
                paper_data.get('pages', ''),
                paper_data.get('citations', 0),
                paper_data.get('abstract', ''),
                paper_data.get('keywords', ''),
                paper_data.get('ranking_type', 'citations'),
                paper_data.get('ranking_value', 0),
                datetime.datetime.now().isoformat(),
                paper_id
            ))
        else:
            # Insert new paper
            self.cursor.execute("""
            INSERT INTO journal_top_papers (
                journal_id, scopus_id, doi, title, authors,
                publication_year, volume, issue, pages,
                citations, abstract, keywords, ranking_type,
                ranking_value, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                journal_id,
                paper_data.get('scopus_id', ''),
                paper_data.get('doi', ''),
                paper_data.get('title', ''),
                paper_data.get('authors', ''),
                paper_data.get('publication_year', 0),
                paper_data.get('volume', ''),
                paper_data.get('issue', ''),
                paper_data.get('pages', ''),
                paper_data.get('citations', 0),
                paper_data.get('abstract', ''),
                paper_data.get('keywords', ''),
                paper_data.get('ranking_type', 'citations'),
                paper_data.get('ranking_value', 0),
                datetime.datetime.now().isoformat()
            ))
            
            paper_id = self.cursor.lastrowid
        
        self.conn.commit()
        return paper_id
    
    def store_similar_journal_paper(self, run_id: str, journal_id: int, paper_data: Dict[str, Any]) -> int:
        """Store a similar paper for a journal.
        
        Args:
            run_id: Run ID
            journal_id: Journal ID
            paper_data: Dictionary with paper information
            
        Returns:
            Paper ID
        """
        # Insert similar paper
        self.cursor.execute("""
        INSERT INTO similar_journal_papers (
            run_id, journal_id, scopus_id, doi, title, authors,
            publication_year, volume, issue, pages,
            citations, abstract, keywords, similarity_score,
            similarity_reason, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            journal_id,
            paper_data.get('scopus_id', ''),
            paper_data.get('doi', ''),
            paper_data.get('title', ''),
            paper_data.get('authors', ''),
            paper_data.get('publication_year', 0),
            paper_data.get('volume', ''),
            paper_data.get('issue', ''),
            paper_data.get('pages', ''),
            paper_data.get('citations', 0),
            paper_data.get('abstract', ''),
            paper_data.get('keywords', ''),
            paper_data.get('similarity_score', 0.0),
            paper_data.get('similarity_reason', ''),
            datetime.datetime.now().isoformat()
        ))
        
        paper_id = self.cursor.lastrowid
        self.conn.commit()
        return paper_id
        
    def get_journal_by_issn(self, issn: str) -> Optional[Dict[str, Any]]:
        """Get journal information by ISSN.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Dictionary with journal information or None if not found
        """
        self.cursor.execute("""
        SELECT * FROM journals
        WHERE issn = ? OR e_issn = ?
        """, (issn, issn))
        
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def get_journal_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get journal information by title.
        
        Args:
            title: Journal title
            
        Returns:
            Dictionary with journal information or None if not found
        """
        self.cursor.execute("""
        SELECT * FROM journals
        WHERE title LIKE ?
        """, (f"%{title}%",))
        
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def get_journal_metrics(self, journal_id: int, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get journal metrics.
        
        Args:
            journal_id: Journal ID
            year: Optional year to filter by
            
        Returns:
            List of dictionaries with journal metrics
        """
        if year:
            self.cursor.execute("""
            SELECT * FROM journal_metrics
            WHERE journal_id = ? AND year = ?
            ORDER BY year DESC
            """, (journal_id, year))
        else:
            self.cursor.execute("""
            SELECT * FROM journal_metrics
            WHERE journal_id = ?
            ORDER BY year DESC
            """, (journal_id,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_journal_top_papers(self, journal_id: int, limit: int = 10, ranking_type: str = 'citations') -> List[Dict[str, Any]]:
        """Get top papers for a journal.
        
        Args:
            journal_id: Journal ID
            limit: Maximum number of papers to return
            ranking_type: Type of ranking (citations, recency, etc.)
            
        Returns:
            List of dictionaries with paper information
        """
        self.cursor.execute(f"""
        SELECT * FROM journal_top_papers
        WHERE journal_id = ? AND ranking_type = ?
        ORDER BY ranking_value DESC
        LIMIT ?
        """, (journal_id, ranking_type, limit))
        
        return [dict(row) for row in self.cursor.fetchall()]
        
    def get_similar_journal_papers(self, run_id: str, journal_id: int) -> List[Dict[str, Any]]:
        """Get similar papers for a journal.
        
        Args:
            run_id: Run ID
            journal_id: Journal ID
            
        Returns:
            List of dictionaries with paper information
        """
        self.cursor.execute("""
        SELECT * FROM similar_journal_papers
        WHERE run_id = ? AND journal_id = ?
        ORDER BY similarity_score DESC
        """, (run_id, journal_id))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_journal_summary(self, journal_id: int) -> Dict[str, Any]:
        """Get a comprehensive summary of a journal.
        
        Args:
            journal_id: Journal ID
            
        Returns:
            Dictionary with journal summary
        """
        # Get journal information
        self.cursor.execute("""
        SELECT * FROM journals
        WHERE journal_id = ?
        """, (journal_id,))
        
        result = self.cursor.fetchone()
        journal = dict(result) if result else {}
        
        # Get latest metrics
        self.cursor.execute("""
        SELECT * FROM journal_metrics
        WHERE journal_id = ?
        ORDER BY year DESC
        LIMIT 1
        """, (journal_id,))
        
        result = self.cursor.fetchone()
        metrics = dict(result) if result else {}
        
        # Get top 5 papers
        self.cursor.execute("""
        SELECT * FROM journal_top_papers
        WHERE journal_id = ?
        ORDER BY citations DESC
        LIMIT 5
        """, (journal_id,))
        
        top_papers = [dict(row) for row in self.cursor.fetchall()]
        
        # Build summary
        summary = {
            "journal": journal,
            "metrics": metrics,
            "top_papers": top_papers
        }
        
        return summary