"""
Constants and shared configurations for the paper revision tool.
"""

from typing import Dict, Any

# Operation modes with their configurations
OPERATION_MODES: Dict[str, Dict[str, Any]] = {
    "training": {
        "description": "Training Mode - Less expensive models, optimized for cost",
        "provider_recommendations": {
            "anthropic": "claude-3-haiku-20240307",
            "openai": "gpt-4o-mini",
            "google": "gemini-1.5-flash"
        },
        "optimize_costs": True,
        "budget": 5.0,
        "competitor_evaluation": False
    },
    "finetuning": {
        "description": "Fine-tuning Mode - Balanced models with moderate quality",
        "provider_recommendations": {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
            "google": "gemini-1.5-pro"
        },
        "optimize_costs": True,
        "budget": 10.0,
        "competitor_evaluation": True
    },
    "final": {
        "description": "Final Mode - Highest quality models, best output quality",
        "provider_recommendations": {
            "anthropic": "claude-opus-4-20250514",
            "openai": "gpt-4.5-preview",
            "google": "gemini-2.5-pro-preview"
        },
        "optimize_costs": False,
        "budget": 20.0,
        "competitor_evaluation": True
    }
}

# Task types for evaluation and optimization
TASK_TYPES = {
    "analysis": "Analysis and review of content",
    "general": "General information processing",
    "text_generation": "Creative text generation",
    "reference_management": "Reference and citation processing",
    "reviewer_analysis": "Analysis of reviewer feedback",
    "editing": "Editing and revision of text",
    "summarization": "Summarizing longer content"
}

# Default output file names
DEFAULT_OUTPUT_FILES = {
    "analysis": "paper_analysis.json",
    "reviewer_analysis": "reviewer_analysis.json",
    "changes_document": "changes_document.docx",
    "revised_paper": "revised_paper.docx",
    "assessment": "assessment.docx",
    "editor_letter": "editor_letter.docx",
    "references": "references.bib",
    "cost_report": "cost_report.txt"
}

# System prompts for different operations
SYSTEM_PROMPTS = {
    "paper_analysis": "You are a scientific paper analysis assistant. Analyze the paper thoroughly and extract key information.",
    "reviewer_analysis": "You are a scientific reviewer analysis assistant. Extract key insights from reviewer comments.",
    "solution_generation": "You are a scientific paper revision assistant. Generate effective solutions to address reviewer concerns.",
    "changes_generation": "You are a scientific paper revision assistant. Generate specific text changes to implement revision solutions.",
    "reference_validation": "You are a scientific reference assistant. Validate and suggest improvements to references.",
    "assessment": "You are a scientific paper assessment assistant. Evaluate the impact of revisions on paper quality.",
    "editor_letter": "You are a scientific communication assistant. Create professional response letters to journal editors."
}

# JSON extraction patterns
JSON_PATTERNS = {
    "start_markers": ["{", "["],
    "end_markers": ["}", "]"],
    "code_block_markers": ["```json", "```"]
}

# Budget thresholds for warnings
BUDGET_THRESHOLDS = {
    "warning": 0.75,  # 75% of budget used - show warning
    "critical": 0.90   # 90% of budget used - show critical warning
}