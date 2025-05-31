"""
Pytest configuration and shared fixtures for the Paper Revision Tool tests.

This module contains shared fixtures that can be used across multiple test modules.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.core.context import RevisionContext
from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.issue import Issue
from src.domain.solution import Solution
from src.domain.change import Change
from src.domain.reference import Reference


# Path helpers
@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir(fixtures_dir) -> Path:
    """Return the path to the test data directory."""
    return fixtures_dir / "data"


@pytest.fixture
def sample_pdf_path(data_dir) -> Path:
    """Return the path to a sample PDF file."""
    return data_dir / "sample_paper.pdf"


@pytest.fixture
def sample_docx_path(data_dir) -> Path:
    """Return the path to a sample DOCX file."""
    return data_dir / "sample_document.docx"


@pytest.fixture
def sample_bibtex_path(data_dir) -> Path:
    """Return the path to a sample BibTeX file."""
    return data_dir / "sample_references.bib"


# Mock API clients
@pytest.fixture
def mock_anthropic_client() -> Mock:
    """Return a mocked Anthropic client."""
    mock = Mock()
    mock.completions.create.return_value.completion = "Mocked Anthropic response"
    return mock


@pytest.fixture
def mock_openai_client() -> Mock:
    """Return a mocked OpenAI client."""
    mock = Mock()
    mock.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Mocked OpenAI response"))
    ]
    return mock


@pytest.fixture
def mock_google_client() -> Mock:
    """Return a mocked Google client."""
    mock = Mock()
    mock.generate_content.return_value.text = "Mocked Google response"
    return mock


# Domain entities
@pytest.fixture
def sample_paper() -> Paper:
    """Return a sample Paper entity."""
    return Paper(
        title="Sample Paper Title",
        authors=["Author One", "Author Two"],
        abstract="This is a sample abstract for testing purposes.",
        sections={
            "Introduction": "This is the introduction section.",
            "Methods": "This is the methods section.",
            "Results": "This is the results section.",
            "Discussion": "This is the discussion section.",
            "Conclusion": "This is the conclusion section."
        },
        references=[
            Reference(key="smith2020", title="Sample Reference 1", authors=["Smith, J."], year=2020),
            Reference(key="jones2021", title="Sample Reference 2", authors=["Jones, A."], year=2021)
        ]
    )


@pytest.fixture
def sample_reviewer_comments() -> list[ReviewerComment]:
    """Return a list of sample ReviewerComment entities."""
    return [
        ReviewerComment(
            reviewer_number=1,
            text="The paper is interesting but needs more clarity in the methods section.",
            overall_assessment="minor_revision",
            main_concerns=["Methods lack clarity", "Missing recent references"],
            methodology_comments=["The methods need more detail"],
            results_comments=["Results are well presented"],
            writing_comments=["The writing is clear but some paragraphs are too long"],
            references_comments=["Missing some recent relevant work"]
        ),
        ReviewerComment(
            reviewer_number=2,
            text="The paper has potential but requires significant improvements.",
            overall_assessment="major_revision",
            main_concerns=["Insufficient analysis", "Weak conclusion"],
            methodology_comments=["The methodology is sound but lacks details"],
            results_comments=["The results need more analysis"],
            writing_comments=["The writing could be improved for clarity"],
            references_comments=["The references are adequate"]
        )
    ]


@pytest.fixture
def sample_issues(sample_paper, sample_reviewer_comments) -> list[Issue]:
    """Return a list of sample Issue entities."""
    return [
        Issue(
            description="Methods section lacks clarity",
            type="methodology",
            severity="high",
            reviewer_number=1,
            section="Methods",
            context="This is the methods section.",
            paper=sample_paper
        ),
        Issue(
            description="Missing recent references",
            type="references",
            severity="medium",
            reviewer_number=1,
            section="References",
            context="References section",
            paper=sample_paper
        ),
        Issue(
            description="Results need more analysis",
            type="results",
            severity="high",
            reviewer_number=2,
            section="Results",
            context="This is the results section.",
            paper=sample_paper
        )
    ]


@pytest.fixture
def sample_solutions(sample_issues) -> list[Solution]:
    """Return a list of sample Solution entities."""
    return [
        Solution(
            description="Improve methods section clarity",
            issue=sample_issues[0],
            approach="Add more details about the methodology",
            impact="High",
            implementation_difficulty="Medium"
        ),
        Solution(
            description="Add missing recent references",
            issue=sample_issues[1],
            approach="Include references from the last 2 years",
            impact="Medium",
            implementation_difficulty="Low"
        ),
        Solution(
            description="Enhance results analysis",
            issue=sample_issues[2],
            approach="Add statistical analysis and interpretation",
            impact="High",
            implementation_difficulty="High"
        )
    ]


@pytest.fixture
def sample_changes(sample_solutions) -> list[Change]:
    """Return a list of sample Change entities."""
    return [
        Change(
            solution=sample_solutions[0],
            section="Methods",
            old_text="This is the methods section.",
            new_text="This is the methods section with improved clarity and additional details about the methodology.",
            reason="To address the reviewer's concern about clarity in the methods section"
        ),
        Change(
            solution=sample_solutions[1],
            section="References",
            old_text="References section",
            new_text="References section with additional recent works from 2022-2024.",
            reason="To address the reviewer's concern about missing recent references"
        ),
        Change(
            solution=sample_solutions[2],
            section="Results",
            old_text="This is the results section.",
            new_text="This is the results section with enhanced statistical analysis and interpretation.",
            reason="To address the reviewer's concern about insufficient analysis in the results"
        )
    ]


# Context fixtures
@pytest.fixture
def test_context() -> RevisionContext:
    """Return a test RevisionContext."""
    context = RevisionContext(
        original_paper_path="/path/to/paper.pdf",
        reviewer_comment_files=["/path/to/reviewer1.txt", "/path/to/reviewer2.txt"],
        output_dir="/path/to/output",
        provider="anthropic",
        model_name="claude-3-opus-20240229"
    )
    
    # Initialize process statistics
    context.process_statistics = {
        "total_tokens": 0,
        "total_cost": 0.0,
        "api_calls": 0,
        "start_time": "2025-01-01T00:00:00Z",
        "steps_completed": []
    }
    
    return context


# Configuration fixtures
@pytest.fixture
def test_config():
    """Return a test configuration."""
    from src.config import AppConfig
    
    # Create a minimal configuration for testing
    config = AppConfig()
    config.llm.provider = "anthropic"
    config.llm.model_name = "claude-3-opus-20240229"
    config.llm.verify_model = False
    
    config.files.original_paper_path = "/path/to/paper.pdf"
    config.files.reviewer_comment_files = ["/path/to/reviewer1.txt", "/path/to/reviewer2.txt"]
    
    config.budget.budget = 10.0
    config.budget.optimize_costs = True
    
    config.output.output_dir = "/path/to/output"
    
    return config


# Mock responses
@pytest.fixture
def mock_anthropic_response():
    """Return a mock Anthropic API response."""
    return {
        "id": "msg_0123456789",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a mock response from the Anthropic API."
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }


@pytest.fixture
def mock_openai_response():
    """Return a mock OpenAI API response."""
    return {
        "id": "chatcmpl-0123456789",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the OpenAI API."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


@pytest.fixture
def mock_google_response():
    """Return a mock Google API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a mock response from the Google API."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "safetyRatings": []
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150
        }
    }