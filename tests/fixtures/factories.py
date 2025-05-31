"""
Test factories for creating test objects.

This module provides factory functions for creating test objects with sensible
defaults. These factories can be used to create objects for testing without
having to specify all the required parameters each time.
"""

import uuid
import random
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment, AssessmentType
from src.domain.issue import Issue, IssueType, IssueSeverity
from src.domain.solution import Solution, ImpactLevel, ImplementationDifficulty
from src.domain.change import Change
from src.domain.reference import Reference


def create_paper(**kwargs) -> Paper:
    """
    Create a Paper object with default values.
    
    Args:
        **kwargs: Override default values
        
    Returns:
        Paper object
    """
    defaults = {
        "title": f"Test Paper {uuid.uuid4().hex[:8]}",
        "authors": ["Test Author One", "Test Author Two"],
        "abstract": "This is a test abstract for testing purposes.",
        "sections": {
            "Introduction": "This is the introduction section.",
            "Methods": "This is the methods section.",
            "Results": "This is the results section.",
            "Discussion": "This is the discussion section.",
            "Conclusion": "This is the conclusion section."
        },
        "references": [create_reference() for _ in range(3)]
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return Paper(**defaults)


def create_reviewer_comment(**kwargs) -> ReviewerComment:
    """
    Create a ReviewerComment object with default values.
    
    Args:
        **kwargs: Override default values
        
    Returns:
        ReviewerComment object
    """
    reviewer_number = kwargs.get("reviewer_number", random.randint(1, 3))
    
    defaults = {
        "reviewer_number": reviewer_number,
        "text": f"This is a test comment from reviewer {reviewer_number}.",
        "overall_assessment": random.choice(list(AssessmentType)),
        "main_concerns": [
            "Test concern one",
            "Test concern two"
        ],
        "methodology_comments": ["The methodology needs improvement"],
        "results_comments": ["The results are well presented"],
        "writing_comments": ["The writing could be clearer"],
        "references_comments": ["Some references are missing"],
        "text_preview": f"Reviewer {reviewer_number} commented on the paper."
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return ReviewerComment(**defaults)


def create_issue(paper: Optional[Paper] = None, **kwargs) -> Issue:
    """
    Create an Issue object with default values.
    
    Args:
        paper: Paper object to associate with the issue
        **kwargs: Override default values
        
    Returns:
        Issue object
    """
    if paper is None:
        paper = create_paper()
    
    section = random.choice(list(paper.sections.keys()))
    
    defaults = {
        "description": f"Test issue in {section} section",
        "type": random.choice(list(IssueType)),
        "severity": random.choice(list(IssueSeverity)),
        "reviewer_number": random.randint(1, 3),
        "section": section,
        "context": paper.sections[section][:100] if section in paper.sections else "Test context",
        "paper": paper
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return Issue(**defaults)


def create_solution(issue: Optional[Issue] = None, **kwargs) -> Solution:
    """
    Create a Solution object with default values.
    
    Args:
        issue: Issue object to associate with the solution
        **kwargs: Override default values
        
    Returns:
        Solution object
    """
    if issue is None:
        issue = create_issue()
    
    defaults = {
        "description": f"Solution for: {issue.description}",
        "issue": issue,
        "approach": "Test approach to solve the issue",
        "impact": random.choice(list(ImpactLevel)),
        "implementation_difficulty": random.choice(list(ImplementationDifficulty))
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return Solution(**defaults)


def create_change(solution: Optional[Solution] = None, **kwargs) -> Change:
    """
    Create a Change object with default values.
    
    Args:
        solution: Solution object to associate with the change
        **kwargs: Override default values
        
    Returns:
        Change object
    """
    if solution is None:
        solution = create_solution()
    
    section = solution.issue.section if solution and solution.issue else "Test Section"
    
    defaults = {
        "solution": solution,
        "section": section,
        "old_text": "This is the old text that needs to be changed.",
        "new_text": "This is the new improved text after the change.",
        "reason": f"To address {solution.issue.description}" if solution and solution.issue else "Test reason"
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return Change(**defaults)


def create_reference(**kwargs) -> Reference:
    """
    Create a Reference object with default values.
    
    Args:
        **kwargs: Override default values
        
    Returns:
        Reference object
    """
    key = f"author{random.randint(2010, 2024)}"
    
    defaults = {
        "key": key,
        "title": f"Test Reference {uuid.uuid4().hex[:8]}",
        "authors": ["Test Author"],
        "year": random.randint(2010, 2024),
        "journal": "Test Journal",
        "volume": str(random.randint(1, 50)),
        "number": str(random.randint(1, 12)),
        "pages": f"{random.randint(1, 100)}--{random.randint(101, 200)}",
        "publisher": "Test Publisher",
        "url": f"https://example.com/{key}",
        "doi": f"10.1234/test.{random.randint(1000, 9999)}"
    }
    
    # Override defaults with provided values
    defaults.update(kwargs)
    
    return Reference(**defaults)


def create_mock_llm_response(provider: str = "anthropic", **kwargs) -> Dict[str, Any]:
    """
    Create a mock LLM response object.
    
    Args:
        provider: LLM provider (anthropic, openai, or google)
        **kwargs: Override default values
        
    Returns:
        Mock LLM response object
    """
    if provider == "anthropic":
        defaults = {
            "id": f"msg_{uuid.uuid4().hex}",
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
                "input_tokens": random.randint(50, 500),
                "output_tokens": random.randint(50, 300)
            }
        }
    elif provider == "openai":
        defaults = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
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
                "prompt_tokens": random.randint(50, 500),
                "completion_tokens": random.randint(50, 300),
                "total_tokens": 0  # Will be calculated
            }
        }
        # Calculate total tokens
        defaults["usage"]["total_tokens"] = defaults["usage"]["prompt_tokens"] + defaults["usage"]["completion_tokens"]
    elif provider == "google":
        defaults = {
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
                "promptTokenCount": random.randint(50, 500),
                "candidatesTokenCount": random.randint(50, 300),
                "totalTokenCount": 0  # Will be calculated
            }
        }
        # Calculate total tokens
        defaults["usageMetadata"]["totalTokenCount"] = (
            defaults["usageMetadata"]["promptTokenCount"] + 
            defaults["usageMetadata"]["candidatesTokenCount"]
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Override defaults with provided values
    if kwargs:
        if provider == "anthropic":
            if "content" in kwargs:
                defaults["content"][0]["text"] = kwargs.pop("content")
        elif provider == "openai":
            if "content" in kwargs:
                defaults["choices"][0]["message"]["content"] = kwargs.pop("content")
        elif provider == "google":
            if "content" in kwargs:
                defaults["candidates"][0]["content"]["parts"][0]["text"] = kwargs.pop("content")
        
        # Update remaining kwargs
        if provider == "anthropic":
            defaults.update(kwargs)
        elif provider == "openai":
            defaults.update(kwargs)
        elif provider == "google":
            defaults.update(kwargs)
    
    return defaults