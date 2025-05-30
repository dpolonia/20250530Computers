#!/usr/bin/env python3
"""
Paper Revision Tool for Computers Journal

This script automates the process of revising a scientific paper based on reviewer comments.
It utilizes LLMs to analyze the original paper, review comments, and generate revision documents.
"""

import os
import sys
import time
import argparse
import datetime
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import click
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model information
try:
    from src.models.openai_models import get_openai_model_choices, get_openai_model_info, get_max_tokens_for_model as get_openai_max_tokens
except ImportError:
    # Create dummy functions if import fails
    def get_openai_model_choices():
        return [
            "gpt-4.5-preview (most powerful)",
            "gpt-4o (powerful)",
            "gpt-4o-mini (balanced)",
            "o1 (powerful reasoning)",
            "o3 (powerful reasoning)",
            "o4-mini (fast reasoning)"
        ]
    
    def get_openai_model_info(model_name):
        return None
    
    def get_openai_max_tokens(model_name):
        return 4096

try:
    from src.models.anthropic_models import get_claude_model_choices, get_claude_model_info, get_max_tokens_for_model as get_claude_max_tokens
except ImportError:
    # Create dummy functions if import fails
    def get_claude_model_choices():
        return [
            "claude-opus-4-20250514 (most powerful)",
            "claude-sonnet-4-20250514 (powerful)",
            "claude-3-7-sonnet-20250219 (powerful)",
            "claude-3-5-sonnet-20241022 (balanced)",
            "claude-3-5-haiku-20241022 (fast)",
            "claude-3-haiku-20240307 (fastest & cheapest)"
        ]
    
    def get_claude_model_info(model_name):
        return None
    
    def get_claude_max_tokens(model_name):
        return 4096
        
try:
    from src.models.google_models import get_gemini_model_choices, get_gemini_model_info, get_max_tokens_for_model as get_gemini_max_tokens
except ImportError:
    # Create dummy functions if import fails
    def get_gemini_model_choices():
        return [
            "gemini-2.5-pro-preview (most powerful, 8M context)",
            "gemini-2.5-flash-preview (efficient, 1M context)",
            "gemini-1.5-pro (powerful, 1M context)",
            "gemini-1.5-flash (fast, 1M context)",
            "gemini-2.0-flash (powerful)",
            "gemini-2.0-flash-lite (faster)"
        ]
    
    def get_gemini_model_info(model_name):
        return None
    
    def get_gemini_max_tokens(model_name):
        return 8192

# Import utility modules
from src.utils.pdf_processor import PDFProcessor
from src.utils.document_processor import DocumentProcessor
from src.utils.reference_validator import ReferenceValidator
from src.utils.llm_client import get_llm_client

# Unified function to get max tokens for any model
def get_max_tokens_for_model(provider, model_name):
    """Get maximum output tokens for a model across any provider."""
    if provider == "openai":
        return get_openai_max_tokens(model_name)
    elif provider == "anthropic":
        return get_claude_max_tokens(model_name)
    elif provider == "google":
        return get_gemini_max_tokens(model_name)
    else:
        # Default fallback
        return 4096

class PaperRevisionTool:
    """Main class for orchestrating the paper revision process."""
    
    def __init__(self, provider: str, model_name: str, debug: bool = False):
        """Initialize the paper revision tool.
        
        Args:
            provider: LLM provider ("anthropic", "openai", or "google")
            model_name: Name of the model to use
            debug: Enable debug mode for additional logging
        """
        self.provider = provider
        self.model_name = model_name
        self.debug = debug
        self.llm_client = None
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Paths
        self.original_paper_path = "./asis/00.pdf"
        self.original_docx_path = "./asis/00.docx"
        self.reviewer1_path = "./asis/01.pdf"
        self.reviewer2_path = "./asis/02.pdf"
        self.reviewer3_path = "./asis/03.pdf"
        self.editor_letter_path = "./asis/04.pdf"
        self.prisma_requirements_path = "./asis/05.pdf"
        self.journal_info_path = "./asis/06.pdf"
        self.scopus_info_paths = ["./asis/07.pdf", "./asis/08.pdf"]
        self.cited_papers_paths = [f"./asis/{i:02d}.pdf" for i in range(11, 19)]
        self.similar_papers_paths = [f"./asis/{i:02d}.pdf" for i in range(21, 25)]
        self.bib_path = "./asis/zz.bib"
        
        # Create model directory inside tobe
        self.model_dir = f"./tobe/{self.model_name.replace(' ', '_').replace('-', '_')}"
        self.timestamp_dir = f"{self.model_dir}/{self.timestamp}"
        
        # Output paths
        self.revision_summary_path = f"{self.timestamp_dir}/90{self.timestamp}.docx"
        self.changes_document_path = f"{self.timestamp_dir}/91{self.timestamp}.docx"
        self.revised_paper_path = f"{self.timestamp_dir}/92{self.timestamp}.docx"
        self.assessment_path = f"{self.timestamp_dir}/93{self.timestamp}.docx"
        self.editor_response_path = f"{self.timestamp_dir}/94{self.timestamp}.docx"
        self.new_bib_path = f"{self.timestamp_dir}/zz{self.timestamp}.bib"
        
        # Ensure directories exist
        os.makedirs("./tobe", exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)
        
        # Statistics
        self.start_time = time.time()
        self.process_statistics = {
            "tokens_used": 0,
            "cost": 0.0,
            "requests": 0,
            "files_processed": 0,
            "files_created": 0
        }
        
        # Initialize LLM client
        self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the LLM client based on selected provider and model."""
        try:
            self.llm_client = get_llm_client(self.provider, self.model_name)
            if not self.llm_client.validate_api_key():
                self._log_error(f"Invalid API key for {self.provider}. Please check your .env file.")
                sys.exit(1)
            self._log_success(f"Successfully initialized {self.provider} client with model {self.model_name}")
        except Exception as e:
            self._log_error(f"Error initializing LLM client: {e}")
            sys.exit(1)
    
    def _log_info(self, message: str):
        """Log an informational message."""
        print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {message}")
    
    def _log_success(self, message: str):
        """Log a success message."""
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")
    
    def _log_warning(self, message: str):
        """Log a warning message."""
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")
    
    def _log_error(self, message: str):
        """Log an error message."""
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")
    
    def _log_debug(self, message: str):
        """Log a debug message."""
        if self.debug:
            print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} {message}")
    
    def _log_stats(self):
        """Log current statistics."""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Update stats from LLM client
        if self.llm_client:
            usage_stats = self.llm_client.get_usage_statistics()
            self.process_statistics["tokens_used"] = usage_stats["total_tokens"]
            self.process_statistics["cost"] = usage_stats["total_cost"]
            self.process_statistics["requests"] = usage_stats["request_count"]
        
        print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}CURRENT STATISTICS:{Style.RESET_ALL}")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"Tokens used: {self.process_statistics['tokens_used']:,}")
        print(f"Estimated cost: ${self.process_statistics['cost']:.4f}")
        print(f"API requests: {self.process_statistics['requests']}")
        print(f"Files processed: {self.process_statistics['files_processed']}")
        print(f"Files created: {self.process_statistics['files_created']}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")
    
    def run(self):
        """Run the full paper revision process."""
        try:
            self._log_info("Starting paper revision process")
            
            # Step 1: Analyze original paper
            self._log_info("Step 1: Analyzing original paper")
            paper_analysis = self._analyze_original_paper()
            
            # Step 2: Analyze reviewer comments
            self._log_info("Step 2: Analyzing reviewer comments")
            reviewer_comments = self._analyze_reviewer_comments()
            
            # Step 3: Process editor letter and PRISMA requirements
            self._log_info("Step 3: Processing editor letter and PRISMA requirements")
            editor_requirements = self._process_editor_requirements()
            
            # Step 4: Analyze journal style and requirements
            self._log_info("Step 4: Analyzing journal style and requirements")
            journal_style = self._analyze_journal_style()
            
            # Step 5: Generate revision summary
            self._log_info("Step 5: Generating revision summary")
            issues, solutions = self._generate_revision_plan(
                paper_analysis, reviewer_comments, editor_requirements, journal_style
            )
            revision_summary_path = self._create_revision_summary(issues, solutions, self.revision_summary_path)
            self._log_success(f"Created revision summary at {revision_summary_path}")
            
            # Step 6: Generate changes document
            self._log_info("Step 6: Generating changes document")
            changes = self._generate_changes(paper_analysis, issues, solutions)
            changes_document_path = self._create_changes_document(changes, self.changes_document_path)
            self._log_success(f"Created changes document at {changes_document_path}")
            
            # Step 7: Validate and update references
            self._log_info("Step 7: Validating and updating references")
            new_references = self._validate_and_update_references(paper_analysis, reviewer_comments, self.new_bib_path)
            
            # Step 8: Create revised paper with track changes
            self._log_info("Step 8: Creating revised paper with track changes")
            revised_paper_path = self._create_revised_paper(changes, self.revised_paper_path)
            self._log_success(f"Created revised paper at {revised_paper_path}")
            
            # Step 9: Create assessment document
            self._log_info("Step 9: Creating assessment document")
            assessment_path = self._create_assessment(changes, paper_analysis, self.assessment_path)
            self._log_success(f"Created assessment document at {assessment_path}")
            
            # Step 10: Create letter to editor
            self._log_info("Step 10: Creating letter to editor")
            editor_letter_path = self._create_editor_letter(reviewer_comments, changes, self.editor_response_path)
            self._log_success(f"Created letter to editor at {editor_letter_path}")
            
            # Final statistics
            self._log_stats()
            self._log_success("Paper revision process completed successfully!")
            
            return {
                "revision_summary": self.revision_summary_path,
                "changes_document": self.changes_document_path,
                "revised_paper": self.revised_paper_path,
                "assessment": self.assessment_path,
                "editor_letter": self.editor_response_path,
                "new_bib": self.new_bib_path
            }
            
        except Exception as e:
            self._log_error(f"Error in paper revision process: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _analyze_original_paper(self) -> Dict[str, Any]:
        """Analyze the original paper.
        
        Returns:
            Dictionary with paper analysis results
        """
        self._log_info("Loading original paper")
        pdf_processor = PDFProcessor(self.original_paper_path)
        self.process_statistics["files_processed"] += 1
        
        sections = pdf_processor.extract_sections()
        tables = pdf_processor.extract_tables()
        figures = pdf_processor.extract_figures()
        references = pdf_processor.extract_references()
        
        # Extract structured information using LLM
        prompt = f"""
        I'm analyzing a scientific paper from the journal Computers (ISSN: 2073-431X).
        I need you to extract and organize key information from this paper.
        
        Here's the paper content:
        
        Title: {sections.get('Title', 'N/A')}
        
        Abstract: {sections.get('Abstract', 'N/A')}
        
        {sections.get('Introduction', 'N/A')}
        
        Provide a structured analysis with the following information:
        1. Title of the paper
        2. Authors
        3. Main research questions or objectives
        4. Methodology used
        5. Key findings
        6. Limitations mentioned
        7. Future work suggested
        8. Overall structure of the paper (list of sections)
        9. Key terms and concepts
        10. Publication details (if available)
        
        Format the response as a JSON object.
        """
        
        self._log_info("Analyzing paper structure and content")
        paper_analysis_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific paper analysis assistant. Extract structured information from papers and format as JSON."
        )
        
        try:
            paper_analysis = json.loads(paper_analysis_json)
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON. Using basic analysis.")
            paper_analysis = {
                "title": sections.get('Title', 'Unknown Title'),
                "authors": "Unknown Authors",
                "objectives": "Unknown Objectives",
                "methodology": "Unknown Methodology",
                "findings": "Unknown Findings",
                "limitations": "Unknown Limitations",
                "future_work": "Unknown Future Work",
                "structure": list(sections.keys()),
                "key_terms": [],
                "publication_details": {}
            }
        
        pdf_processor.close()
        
        # Add sections, tables, and figures to the analysis
        paper_analysis["sections"] = sections
        paper_analysis["tables"] = tables
        paper_analysis["figures"] = [caption for caption, _ in figures]
        paper_analysis["references"] = references
        
        self._log_success("Paper analysis completed")
        return paper_analysis
    
    def _analyze_reviewer_comments(self) -> List[Dict[str, Any]]:
        """Analyze the reviewer comments.
        
        Returns:
            List of dictionaries with reviewer comment analysis
        """
        reviewer_paths = [self.reviewer1_path, self.reviewer2_path, self.reviewer3_path]
        reviewer_comments = []
        
        for i, path in enumerate(reviewer_paths, 1):
            self._log_info(f"Analyzing reviewer {i} comments")
            pdf_processor = PDFProcessor(path)
            self.process_statistics["files_processed"] += 1
            
            text = pdf_processor.text
            
            prompt = f"""
            I'm analyzing reviewer comments for a scientific paper from the journal Computers (ISSN: 2073-431X).
            Please extract and categorize the reviewer's comments.
            
            Here are the reviewer comments:
            
            {text[:10000]}  # Limit text length to avoid token limits
            
            Provide a structured analysis with the following:
            1. Overall assessment (positive, neutral, negative)
            2. Main concerns
            3. Required changes (must be addressed)
            4. Suggested changes (optional)
            5. Comments on methodology
            6. Comments on results
            7. Comments on writing/presentation
            8. Comments on references/citations
            
            Format the response as a JSON object.
            """
            
            reviewer_analysis_json = self.llm_client.get_completion(
                prompt=prompt,
                system_prompt=f"You are a scientific reviewer analysis assistant. Analyze reviewer {i}'s comments and format as JSON."
            )
            
            try:
                reviewer_analysis = json.loads(reviewer_analysis_json)
            except json.JSONDecodeError:
                # Fallback if LLM didn't return valid JSON
                self._log_warning(f"LLM didn't return valid JSON for reviewer {i}. Using basic analysis.")
                reviewer_analysis = {
                    "overall_assessment": "Unknown",
                    "main_concerns": ["Unknown concerns"],
                    "required_changes": ["Unknown required changes"],
                    "suggested_changes": ["Unknown suggested changes"],
                    "methodology_comments": ["Unknown methodology comments"],
                    "results_comments": ["Unknown results comments"],
                    "writing_comments": ["Unknown writing comments"],
                    "references_comments": ["Unknown references comments"]
                }
            
            # Add full text and reviewer number
            reviewer_analysis["full_text"] = text
            reviewer_analysis["reviewer_number"] = i
            
            reviewer_comments.append(reviewer_analysis)
            pdf_processor.close()
        
        self._log_success("Reviewer comment analysis completed")
        return reviewer_comments
    
    def _process_editor_requirements(self) -> Dict[str, Any]:
        """Process editor letter and PRISMA requirements.
        
        Returns:
            Dictionary with editor requirements
        """
        # Process editor letter
        self._log_info("Processing editor letter")
        editor_pdf = PDFProcessor(self.editor_letter_path)
        self.process_statistics["files_processed"] += 1
        editor_text = editor_pdf.text
        editor_pdf.close()
        
        # Process PRISMA requirements
        self._log_info("Processing PRISMA requirements")
        prisma_pdf = PDFProcessor(self.prisma_requirements_path)
        self.process_statistics["files_processed"] += 1
        prisma_text = prisma_pdf.text
        prisma_pdf.close()
        
        prompt = f"""
        I'm analyzing the editor letter and PRISMA requirements for a scientific paper revision.
        
        Editor Letter:
        {editor_text[:5000]}  # Limit text length to avoid token limits
        
        PRISMA Requirements:
        {prisma_text[:5000]}  # Limit text length to avoid token limits
        
        Provide a structured analysis with the following:
        1. Editor's main requirements
        2. Editor's suggested changes
        3. Editor's decision (reject, revise, accept with revisions, etc.)
        4. PRISMA framework requirements
        5. How the paper should be modified to meet PRISMA requirements
        
        Format the response as a JSON object.
        """
        
        editor_analysis_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific editor analysis assistant. Extract requirements from editor letters and PRISMA guidelines."
        )
        
        try:
            editor_requirements = json.loads(editor_analysis_json)
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON for editor requirements. Using basic analysis.")
            editor_requirements = {
                "editor_requirements": ["Unknown requirements"],
                "editor_suggestions": ["Unknown suggestions"],
                "editor_decision": "Unknown decision",
                "prisma_requirements": ["Unknown PRISMA requirements"],
                "prisma_modifications": ["Unknown PRISMA modifications"]
            }
        
        # Add full texts
        editor_requirements["editor_letter_text"] = editor_text
        editor_requirements["prisma_text"] = prisma_text
        
        self._log_success("Editor requirements analysis completed")
        return editor_requirements
    
    def _analyze_journal_style(self) -> Dict[str, Any]:
        """Analyze journal style and requirements.
        
        Returns:
            Dictionary with journal style information
        """
        # Process journal information
        self._log_info("Processing journal information")
        journal_pdf = PDFProcessor(self.journal_info_path)
        self.process_statistics["files_processed"] += 1
        journal_text = journal_pdf.text
        journal_pdf.close()
        
        # Process Scopus information
        scopus_text = ""
        for path in self.scopus_info_paths:
            self._log_info(f"Processing Scopus information from {path}")
            scopus_pdf = PDFProcessor(path)
            self.process_statistics["files_processed"] += 1
            scopus_text += scopus_pdf.text + "\n\n"
            scopus_pdf.close()
        
        # Process highly cited papers for style analysis
        self._log_info("Processing highly cited papers for style analysis")
        cited_papers_sample = []
        for i, path in enumerate(self.cited_papers_paths[:3]):  # Process only first 3 to save time
            cited_pdf = PDFProcessor(path)
            self.process_statistics["files_processed"] += 1
            sections = cited_pdf.extract_sections()
            # Get a sample of sections
            sample = {
                "paper_number": i + 1,
                "title": sections.get("Title", "Unknown Title"),
                "abstract": sections.get("Abstract", "Unknown Abstract")[:500],  # Limit length
                "introduction": sections.get("Introduction", "Unknown Introduction")[:500]  # Limit length
            }
            cited_papers_sample.append(sample)
            cited_pdf.close()
        
        # Process similar papers
        self._log_info("Processing similar papers for style analysis")
        similar_papers_sample = []
        for i, path in enumerate(self.similar_papers_paths[:2]):  # Process only first 2 to save time
            similar_pdf = PDFProcessor(path)
            self.process_statistics["files_processed"] += 1
            sections = similar_pdf.extract_sections()
            # Get a sample of sections
            sample = {
                "paper_number": i + 1,
                "title": sections.get("Title", "Unknown Title"),
                "abstract": sections.get("Abstract", "Unknown Abstract")[:500],  # Limit length
                "introduction": sections.get("Introduction", "Unknown Introduction")[:500]  # Limit length
            }
            similar_papers_sample.append(sample)
            similar_pdf.close()
        
        # Analyze reference style
        self._log_info("Analyzing reference style")
        ref_validator = ReferenceValidator(self.bib_path)
        self.process_statistics["files_processed"] += 1
        citation_style = ref_validator.get_citation_style()
        
        # Combine all information and use LLM to extract style guidelines
        prompt = f"""
        I'm analyzing the style requirements for a scientific paper to be published in the journal Computers (ISSN: 2073-431X).
        
        Journal Information:
        {journal_text[:3000]}  # Limit text length
        
        Scopus Information:
        {scopus_text[:3000]}  # Limit text length
        
        Citation Style:
        {citation_style}
        
        Sample of Highly Cited Papers from this Journal:
        {json.dumps(cited_papers_sample, indent=2)}
        
        Sample of Similar Papers to the Current Paper:
        {json.dumps(similar_papers_sample, indent=2)}
        
        Based on this information, provide a comprehensive style guide for this journal, including:
        1. General formatting requirements
        2. Section structure
        3. Figure and table presentation
        4. Citation and reference style
        5. Language and tone
        6. Common elements in highly cited papers
        7. Typical structure for papers similar to the current one
        
        Format the response as a JSON object.
        """
        
        style_analysis_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific journal style analysis assistant. Extract style guidelines from journal information and sample papers."
        )
        
        try:
            journal_style = json.loads(style_analysis_json)
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON for journal style. Using basic analysis.")
            journal_style = {
                "formatting": ["Unknown formatting requirements"],
                "section_structure": ["Unknown section structure"],
                "figures_and_tables": ["Unknown figure and table guidelines"],
                "citation_style": citation_style,
                "language_and_tone": ["Unknown language and tone guidelines"],
                "successful_papers": ["Unknown successful paper elements"],
                "similar_papers_structure": ["Unknown structure for similar papers"]
            }
        
        self._log_success("Journal style analysis completed")
        return journal_style
    
    def _generate_revision_plan(
        self,
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        editor_requirements: Dict[str, Any],
        journal_style: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate a revision plan based on all analyses.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            editor_requirements: Editor and PRISMA requirements
            journal_style: Journal style guidelines
            
        Returns:
            Tuple of (issues, solutions) as lists of dictionaries
        """
        self._log_info("Generating revision plan")
        
        # Prepare input for LLM
        paper_summary = {
            "title": paper_analysis.get("title", "Unknown Title"),
            "objectives": paper_analysis.get("objectives", "Unknown Objectives"),
            "methodology": paper_analysis.get("methodology", "Unknown Methodology"),
            "findings": paper_analysis.get("findings", "Unknown Findings"),
            "structure": paper_analysis.get("structure", [])
        }
        
        reviewer_summary = []
        for i, reviewer in enumerate(reviewer_comments, 1):
            main_concerns = reviewer.get("main_concerns", [])
            if main_concerns is None:
                main_concerns = []
            required_changes = reviewer.get("required_changes", [])
            if required_changes is None:
                required_changes = []
                
            summary = {
                "reviewer": i,
                "assessment": reviewer.get("overall_assessment", "Unknown"),
                "main_concerns": main_concerns[:3] if len(main_concerns) > 0 else [],  # Limit to top 3
                "required_changes": required_changes[:3] if len(required_changes) > 0 else []  # Limit to top 5
            }
            reviewer_summary.append(summary)
        
        editor_reqs = editor_requirements.get("editor_requirements", [])
        if editor_reqs is None:
            editor_reqs = []
        prisma_reqs = editor_requirements.get("prisma_requirements", [])
        if prisma_reqs is None:
            prisma_reqs = []
            
        editor_summary = {
            "requirements": editor_reqs[:5] if len(editor_reqs) > 0 else [],  # Limit to top 5
            "decision": editor_requirements.get("editor_decision", "Unknown"),
            "prisma_requirements": prisma_reqs[:5] if len(prisma_reqs) > 0 else []  # Limit to top 5
        }
        
        formatting = journal_style.get("formatting", [])
        if formatting is None:
            formatting = []
        section_structure = journal_style.get("section_structure", [])
        if section_structure is None:
            section_structure = []
            
        style_summary = {
            "formatting": formatting[:3] if len(formatting) > 0 else [],  # Limit to top 3
            "section_structure": section_structure[:3] if len(section_structure) > 0 else [],  # Limit to top 3
            "citation_style": journal_style.get("citation_style", "Unknown")
        }
        
        prompt = f"""
        I'm developing a revision plan for a scientific paper based on reviewer comments, editor requirements, and journal style guidelines.
        
        Paper Summary:
        {json.dumps(paper_summary, indent=2)}
        
        Reviewer Comments:
        {json.dumps(reviewer_summary, indent=2)}
        
        Editor Requirements:
        {json.dumps(editor_summary, indent=2)}
        
        Journal Style:
        {json.dumps(style_summary, indent=2)}
        
        Based on this information, generate:
        
        1. A list of issues that need to be addressed, with each issue having:
           - title (short description)
           - description (detailed explanation)
           - source (reviewer number, editor, or style guidelines)
           - severity (critical, major, minor)
        
        2. A list of solutions to address these issues, with each solution having:
           - title (short description)
           - implementation (detailed explanation of changes needed)
           - complexity (high, medium, low)
           - impact (how this addresses the issues)
        
        Ensure that each solution addresses one or more of the identified issues.
        Format the response as a JSON object with "issues" and "solutions" arrays.
        """
        
        revision_plan_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific paper revision assistant. Generate structured revision plans based on reviewer comments and requirements."
        )
        
        try:
            revision_plan = json.loads(revision_plan_json)
            issues = revision_plan.get("issues", [])
            solutions = revision_plan.get("solutions", [])
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON for revision plan. Using basic plan.")
            issues = []
            for i, reviewer in enumerate(reviewer_comments):
                main_concerns = reviewer.get("main_concerns", ["Unknown concerns"])
                if main_concerns is None:
                    main_concerns = ["Unknown concerns"]
                
                # Safely slice the list, ensuring it's not None
                top_concerns = main_concerns[:2] if len(main_concerns) >= 2 else main_concerns
                
                issues.append({
                    "title": f"Reviewer {i+1} concern",
                    "description": ", ".join(top_concerns),
                    "source": f"Reviewer {i+1}",
                    "severity": "major"
                })
            
            solutions = [{
                "title": f"Address reviewer {i+1} concerns",
                "implementation": "Review and implement the changes requested by the reviewer.",
                "complexity": "medium",
                "impact": "Addresses the main concerns of the reviewer."
            } for i in range(len(reviewer_comments))]
        
        self._log_success(f"Generated revision plan with {len(issues)} issues and {len(solutions)} solutions")
        return issues, solutions
    
    def _create_revision_summary(
        self,
        issues: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Create a revision summary document.
        
        Args:
            issues: List of issues to address
            solutions: List of solutions
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        self._log_info("Creating revision summary document")
        
        # Load the original docx if it exists, otherwise use a new document
        if os.path.exists(self.original_docx_path):
            doc_processor = DocumentProcessor(self.original_docx_path)
        else:
            # Create a new document from the PDF
            pdf_processor = PDFProcessor(self.original_paper_path)
            temp_docx_path = pdf_processor.pdf_to_docx()
            pdf_processor.close()
            doc_processor = DocumentProcessor(temp_docx_path)
        
        # Create the revision summary document
        summary_path = doc_processor.create_revision_summary(issues, solutions, output_path)
        self.process_statistics["files_created"] += 1
        
        return summary_path
    
    def _generate_changes(
        self,
        paper_analysis: Dict[str, Any],
        issues: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str, Optional[int]]]:
        """Generate specific text changes based on the revision plan.
        
        Args:
            paper_analysis: Analysis of the original paper
            issues: List of issues to address
            solutions: List of solutions
            
        Returns:
            List of tuples (old_text, new_text, reason, line_number)
        """
        self._log_info("Generating specific text changes")
        
        # Prepare input for LLM
        paper_sections = paper_analysis.get("sections", {})
        section_samples = {}
        for section_name, content in paper_sections.items():
            # Limit section content to save tokens
            section_samples[section_name] = content[:1000] + "..." if len(content) > 1000 else content
        
        solutions_summary = []
        for solution in solutions:
            summary = {
                "title": solution.get("title", "Unknown solution"),
                "implementation": solution.get("implementation", "Unknown implementation"),
                "complexity": solution.get("complexity", "medium")
            }
            solutions_summary.append(summary)
        
        prompt = f"""
        I'm generating specific text changes for a scientific paper revision.
        
        Paper Sections:
        {json.dumps(section_samples, indent=2)}
        
        Solutions to Implement:
        {json.dumps(solutions_summary, indent=2)}
        
        Based on this information, generate a list of specific text changes that should be made to implement the solutions.
        For each change, provide:
        1. The original text to be replaced
        2. The new text to replace it with
        3. The reason for the change
        4. The approximate line number (can be an estimate)
        
        Focus on substantive changes that address the revision requirements. Include:
        - Changes to improve methodology description
        - Changes to address limitations
        - Changes to improve the presentation of results
        - Changes to improve the structure and flow
        - Changes to address specific reviewer concerns
        
        Generate at least 5 but no more than 15 specific changes.
        Format the response as a JSON array of objects with "old_text", "new_text", "reason", and "line_number" fields.
        """
        
        changes_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific paper revision assistant. Generate specific text changes to implement revision solutions."
        )
        
        try:
            changes_data = json.loads(changes_json)
            # Convert to list of tuples
            changes = [(
                item.get("old_text", ""),
                item.get("new_text", ""),
                item.get("reason", ""),
                item.get("line_number")
            ) for item in changes_data]
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON for changes. Using basic changes.")
            changes = []
            for section_name, content in list(section_samples.items())[:3]:  # Use first 3 sections
                if len(content) > 100:
                    changes.append((
                        content[:100],
                        f"[REVISED] {content[:100]}",
                        f"Improve {section_name} section based on reviewer comments",
                        None
                    ))
        
        self._log_success(f"Generated {len(changes)} specific text changes")
        return changes
    
    def _create_changes_document(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """Create a document detailing all changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        self._log_info("Creating changes document")
        
        # Load the original docx if it exists, otherwise use a new document
        if os.path.exists(self.original_docx_path):
            doc_processor = DocumentProcessor(self.original_docx_path)
        else:
            # Create a new document from the PDF
            pdf_processor = PDFProcessor(self.original_paper_path)
            temp_docx_path = pdf_processor.pdf_to_docx()
            pdf_processor.close()
            doc_processor = DocumentProcessor(temp_docx_path)
        
        # Create the changes document
        changes_path = doc_processor.create_changes_document(changes, output_path)
        self.process_statistics["files_created"] += 1
        
        return changes_path
    
    def _validate_and_update_references(
        self,
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, str]]:
        """Validate existing references and add new ones based on reviewer suggestions.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            output_path: Path where the new BibTeX file should be saved
            
        Returns:
            List of new references added
        """
        self._log_info("Validating and updating references")
        
        # Load references
        ref_validator = ReferenceValidator(self.bib_path)
        
        # Validate existing references
        valid_refs, invalid_refs = ref_validator.validate_references()
        self._log_info(f"Found {len(valid_refs)} valid and {len(invalid_refs)} invalid references")
        
        # Extract reference suggestions from reviewer comments
        reference_comments = []
        for reviewer in reviewer_comments:
            ref_comments = reviewer.get("references_comments", [])
            reference_comments.extend(ref_comments)
        
        # Generate new references based on reviewer suggestions
        if reference_comments:
            prompt = f"""
            I'm updating references for a scientific paper based on reviewer comments.
            
            Current Paper References:
            {paper_analysis.get('references', ['No references available'])[:10]}
            
            Reviewer Comments on References:
            {reference_comments}
            
            Based on these comments, suggest new references that should be added to the paper.
            For each reference, provide:
            1. Title
            2. Authors
            3. Journal/Conference
            4. Year
            5. DOI (if you can estimate it)
            6. Why this reference should be added
            
            Format the response as a JSON array of objects with the fields above.
            """
            
            new_refs_json = self.llm_client.get_completion(
                prompt=prompt,
                system_prompt="You are a scientific reference assistant. Suggest new references based on reviewer comments."
            )
            
            try:
                new_refs_data = json.loads(new_refs_json)
                new_references = []
                
                for ref_data in new_refs_data:
                    # Convert to BibTeX entry format
                    entry = {
                        "ENTRYTYPE": "article",
                        "title": ref_data.get("title", "Unknown Title"),
                        "author": ref_data.get("authors", "Unknown Authors"),
                        "journal": ref_data.get("journal", ref_data.get("conference", "Unknown Venue")),
                        "year": str(ref_data.get("year", "2023")),
                        "doi": ref_data.get("doi", "")
                    }
                    
                    # Add to reference validator
                    ref_id = ref_validator.add_reference(entry)
                    
                    new_references.append({
                        "id": ref_id,
                        "title": entry["title"],
                        "authors": entry["author"],
                        "year": entry["year"],
                        "reason": ref_data.get("why", "Suggested by reviewer")
                    })
                
                # Save updated references
                ref_validator.export_references(
                    set(list(valid_refs) + [ref["id"] for ref in new_references]),
                    output_path
                )
                self.process_statistics["files_created"] += 1
                
                self._log_success(f"Added {len(new_references)} new references")
                return new_references
                
            except json.JSONDecodeError:
                self._log_warning("LLM didn't return valid JSON for new references. No new references added.")
        
        # If no new references, just export valid ones
        ref_validator.export_references(valid_refs, output_path)
        self.process_statistics["files_created"] += 1
        
        return []
    
    def _create_revised_paper(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """Create revised paper with track changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        self._log_info("Creating revised paper with track changes")
        
        # Load the original docx
        if os.path.exists(self.original_docx_path):
            doc_processor = DocumentProcessor(self.original_docx_path)
        else:
            # Create a new document from the PDF
            pdf_processor = PDFProcessor(self.original_paper_path)
            temp_docx_path = pdf_processor.pdf_to_docx()
            pdf_processor.close()
            doc_processor = DocumentProcessor(temp_docx_path)
        
        # Apply changes with track changes
        changes_applied = 0
        for old_text, new_text, reason, _ in changes:
            if doc_processor.add_tracked_change(old_text, new_text, reason):
                changes_applied += 1
        
        # Save the revised document
        doc_processor.save(output_path)
        self.process_statistics["files_created"] += 1
        
        self._log_success(f"Applied {changes_applied} changes to the paper")
        return output_path
    
    def _create_assessment(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        paper_analysis: Dict[str, Any],
        output_path: str
    ) -> str:
        """Create assessment document.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            paper_analysis: Analysis of the original paper
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        self._log_info("Creating assessment document")
        
        # Create a new document
        from docx import Document
        doc = Document()
        
        # Add heading
        doc.add_heading('Revision Assessment', 0)
        
        # Add introduction
        doc.add_paragraph(
            'This document assesses the changes made to the paper in response to reviewer comments. ' +
            'It evaluates the impact of these changes and identifies any issues that may need to be addressed manually.'
        )
        
        # Summarize changes
        doc.add_heading('Summary of Changes', 1)
        
        changes_by_type = {}
        for _, _, reason, _ in changes:
            change_type = reason.split(' ')[0] if reason else 'Other'
            if change_type not in changes_by_type:
                changes_by_type[change_type] = 0
            changes_by_type[change_type] += 1
        
        p = doc.add_paragraph('Changes by type:')
        for change_type, count in changes_by_type.items():
            p = doc.add_paragraph(f'{change_type}: {count} changes', style='List Bullet')
        
        # Impact assessment
        doc.add_heading('Impact Assessment', 1)
        
        # Use LLM to generate impact assessment
        paper_summary = {
            "title": paper_analysis.get("title", "Unknown Title"),
            "objectives": paper_analysis.get("objectives", "Unknown Objectives"),
            "methodology": paper_analysis.get("methodology", "Unknown Methodology"),
            "findings": paper_analysis.get("findings", "Unknown Findings")
        }
        
        changes_summary = []
        for old_text, new_text, reason, _ in changes[:5]:  # Use first 5 changes as examples
            summary = {
                "reason": reason,
                "old_text_preview": old_text[:50] + "..." if len(old_text) > 50 else old_text,
                "new_text_preview": new_text[:50] + "..." if len(new_text) > 50 else new_text
            }
            changes_summary.append(summary)
        
        prompt = f"""
        I'm assessing the impact of revisions made to a scientific paper.
        
        Paper Summary:
        {json.dumps(paper_summary, indent=2)}
        
        Sample of Changes Made:
        {json.dumps(changes_summary, indent=2)}
        
        Total changes: {len(changes)}
        
        Provide an assessment of:
        1. Overall impact of the changes on the paper's quality
        2. How well the changes address likely reviewer concerns
        3. Potential remaining issues that might need manual attention
        4. Specific areas where the paper has been strengthened
        5. Recommendations for any final manual adjustments
        
        Format the response as a JSON object with these five sections.
        """
        
        assessment_json = self.llm_client.get_completion(
            prompt=prompt,
            system_prompt="You are a scientific paper assessment assistant. Evaluate the impact of revisions on paper quality."
        )
        
        try:
            assessment = json.loads(assessment_json)
            
            # Add overall impact
            doc.add_heading('Overall Impact', 2)
            doc.add_paragraph(assessment.get("overall_impact", "No assessment available."))
            
            # Add reviewer concerns addressed
            doc.add_heading('Reviewer Concerns Addressed', 2)
            doc.add_paragraph(assessment.get("reviewer_concerns", "No assessment available."))
            
            # Add remaining issues
            doc.add_heading('Remaining Issues', 2)
            remaining_issues = assessment.get("remaining_issues", "No issues identified.")
            if isinstance(remaining_issues, list):
                for issue in remaining_issues:
                    doc.add_paragraph(issue, style='List Bullet')
            else:
                doc.add_paragraph(remaining_issues)
            
            # Add strengthened areas
            doc.add_heading('Strengthened Areas', 2)
            strengthened_areas = assessment.get("strengthened_areas", "No areas identified.")
            if isinstance(strengthened_areas, list):
                for area in strengthened_areas:
                    doc.add_paragraph(area, style='List Bullet')
            else:
                doc.add_paragraph(strengthened_areas)
            
            # Add recommendations
            doc.add_heading('Recommendations', 2)
            recommendations = assessment.get("recommendations", "No recommendations.")
            if isinstance(recommendations, list):
                for rec in recommendations:
                    doc.add_paragraph(rec, style='List Bullet')
            else:
                doc.add_paragraph(recommendations)
            
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            self._log_warning("LLM didn't return valid JSON for assessment. Using basic assessment.")
            
            doc.add_heading('Overall Impact', 2)
            doc.add_paragraph('The changes have addressed many of the issues identified in the review process.')
            
            doc.add_heading('Remaining Tasks', 2)
            doc.add_paragraph('The following tasks should be completed manually:')
            doc.add_paragraph('1. Review all changes for accuracy and consistency.', style='List Bullet')
            doc.add_paragraph('2. Check references for proper formatting.', style='List Bullet')
            doc.add_paragraph('3. Ensure all reviewer comments have been addressed.', style='List Bullet')
        
        # Add conclusion
        doc.add_heading('Conclusion', 1)
        doc.add_paragraph(
            'This assessment provides an overview of the changes made and their impact. ' +
            'Authors should review the revised paper carefully before submission.'
        )
        
        # Save the document
        doc.save(output_path)
        self.process_statistics["files_created"] += 1
        
        self._log_success("Created assessment document")
        return output_path
    
    def _create_editor_letter(
        self,
        reviewer_comments: List[Dict[str, Any]],
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """Create letter to the editor.
        
        Args:
            reviewer_comments: Analysis of reviewer comments
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        self._log_info("Creating letter to editor")
        
        # Process changes to create responses to reviewers
        reviewer_responses = []
        
        for i, reviewer in enumerate(reviewer_comments, 1):
            # Group changes by reason to identify which ones address this reviewer's comments
            changes_by_reason = {}
            for old_text, new_text, reason, _ in changes:
                if f"Reviewer {i}" in reason or f"reviewer {i}" in reason.lower():
                    if reason not in changes_by_reason:
                        changes_by_reason[reason] = []
                    changes_by_reason[reason].append((old_text, new_text))
            
            # Extract main comments from the reviewer
            main_concerns = reviewer.get("main_concerns", [])
            required_changes = reviewer.get("required_changes", [])
            
            # Create a list of comment-response pairs
            comments = []
            
            # Add responses to main concerns
            for j, concern in enumerate(main_concerns, 1):
                response_text = "We appreciate this concern and have addressed it in our revision."
                changes_text = "No specific changes were made for this comment."
                
                # Find changes related to this concern
                for reason, change_list in changes_by_reason.items():
                    if any(keyword in concern.lower() for keyword in reason.lower().split()):
                        changes_text = f"We made {len(change_list)} changes to address this concern, including: "
                        changes_text += ", ".join([f"replacing '{old[:20]}...' with '{new[:20]}...'" 
                                                 for old, new in change_list[:2]])
                        if len(change_list) > 2:
                            changes_text += f", and {len(change_list) - 2} more changes."
                
                comments.append({
                    "comment": concern,
                    "response": response_text,
                    "changes": changes_text
                })
            
            # Add responses to required changes
            for j, required in enumerate(required_changes, 1):
                response_text = "We have implemented this required change in our revision."
                changes_text = "No specific changes were made for this comment."
                
                # Find changes related to this requirement
                for reason, change_list in changes_by_reason.items():
                    if any(keyword in required.lower() for keyword in reason.lower().split()):
                        changes_text = f"We made {len(change_list)} changes to address this requirement, including: "
                        changes_text += ", ".join([f"replacing '{old[:20]}...' with '{new[:20]}...'" 
                                                 for old, new in change_list[:2]])
                        if len(change_list) > 2:
                            changes_text += f", and {len(change_list) - 2} more changes."
                
                comments.append({
                    "comment": required,
                    "response": response_text,
                    "changes": changes_text
                })
            
            reviewer_responses.append({
                "reviewer": i,
                "comments": comments
            })
        
        # Create the editor letter
        doc_processor = DocumentProcessor(self.original_docx_path) if os.path.exists(self.original_docx_path) else None
        
        if doc_processor:
            editor_letter_path = doc_processor.create_editor_letter(reviewer_responses, output_path)
        else:
            # Create manually if original docx not available
            from docx import Document
            doc = Document()
            
            # Add header information
            doc.add_paragraph(f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}")
            doc.add_paragraph("To: The Editor, Computers (ISSN: 2073-431X)")
            doc.add_paragraph("Subject: Revised manuscript submission")
            doc.add_paragraph()
            
            # Add salutation
            doc.add_paragraph("Dear Editor,")
            
            # Add introduction
            doc.add_paragraph(
                "Thank you for the opportunity to revise our manuscript. We have carefully addressed all the comments " +
                "provided by the reviewers and made the necessary changes to improve the quality of our paper. " +
                "We believe that the revised version addresses all the concerns raised and significantly improves the manuscript."
            )
            
            # Add reviewer responses
            doc.add_heading('Responses to Reviewer Comments', 1)
            
            for i, response in enumerate(reviewer_responses, 1):
                doc.add_heading(f"Reviewer {i}", 2)
                
                for j, comment in enumerate(response['comments'], 1):
                    p = doc.add_paragraph()
                    p.add_run(f"Comment {j}: ").bold = True
                    p.add_run(comment['comment'])
                    
                    p = doc.add_paragraph()
                    p.add_run("Response: ").bold = True
                    p.add_run(comment['response'])
                    
                    p = doc.add_paragraph()
                    p.add_run("Changes made: ").bold = True
                    p.add_run(comment['changes'])
                    
                    doc.add_paragraph()
            
            # Add closing
            doc.add_paragraph(
                "We hope that the revised manuscript now meets the standards for publication in Computers. " +
                "We look forward to your feedback and are available to address any additional questions or concerns."
            )
            
            doc.add_paragraph("Sincerely,")
            doc.add_paragraph("The Authors")
            
            # Save the document
            doc.save(output_path)
            editor_letter_path = output_path
        
        self.process_statistics["files_created"] += 1
        
        self._log_success("Created letter to editor")
        return editor_letter_path

def choose_model():
    """Interactive model selection."""
    colorama_init()
    
    print(f"{Fore.CYAN}Choose LLM Provider:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}1.{Style.RESET_ALL} Anthropic Claude")
    print(f"{Fore.CYAN}2.{Style.RESET_ALL} OpenAI GPT")
    print(f"{Fore.CYAN}3.{Style.RESET_ALL} Google Gemini")
    
    provider_choice = input("Enter choice (1-3): ")
    
    if provider_choice == "1":
        provider = "anthropic"
        models = get_claude_model_choices()
    elif provider_choice == "2":
        provider = "openai"
        models = get_openai_model_choices()
    elif provider_choice == "3":
        provider = "google"
        models = get_gemini_model_choices()
    else:
        print(f"{Fore.RED}Invalid choice. Defaulting to Anthropic Claude.{Style.RESET_ALL}")
        provider = "anthropic"
        models = get_claude_model_choices()
    
    print(f"\n{Fore.CYAN}Choose Model:{Style.RESET_ALL}")
    for i, model in enumerate(models, 1):
        print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} {model}")
    
    model_choice = input(f"Enter choice (1-{len(models)}): ")
    
    try:
        model_index = int(model_choice) - 1
        if 0 <= model_index < len(models):
            chosen_model = models[model_index]
        else:
            print(f"{Fore.RED}Invalid choice. Defaulting to first model.{Style.RESET_ALL}")
            chosen_model = models[0]
    except ValueError:
        print(f"{Fore.RED}Invalid input. Defaulting to first model.{Style.RESET_ALL}")
        chosen_model = models[0]
    
    return provider, chosen_model

def main():
    """Main entry point for the paper revision tool."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Paper Revision Tool for Computers Journal")
    parser.add_argument("--provider", choices=["anthropic", "openai", "google"], help="LLM provider")
    parser.add_argument("--model", help="Model name (specific to provider)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Choose model interactively if not specified via command line
    if not args.provider or not args.model:
        provider, model = choose_model()
    else:
        provider = args.provider
        model = args.model
    
    # Create and run the paper revision tool
    revision_tool = PaperRevisionTool(provider, model, args.debug)
    results = revision_tool.run()
    
    if results:
        print(f"\n{Fore.GREEN}Paper revision completed successfully!{Style.RESET_ALL}")
        print("Output files:")
        for name, path in results.items():
            print(f"- {name}: {path}")
    else:
        print(f"\n{Fore.RED}Paper revision failed.{Style.RESET_ALL}")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()