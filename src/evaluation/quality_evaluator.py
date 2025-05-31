"""
Quality evaluator module for assessing the quality of model responses.
"""

import logging
import re
import json
from typing import Dict, Any, Optional

from src.core.context import RevisionContext
from src.core.json_utils import parse_json_safely
from src.utils.llm_client import get_llm_client


class QualityEvaluator:
    """
    Evaluates the quality of model responses.
    
    This class is responsible for assessing the quality of responses from
    language models, both algorithmically and using competing models.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the quality evaluator.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def evaluate_response_quality(
        self, 
        prompt: str, 
        response: str, 
        task_type: str = "general",
        use_competitor: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a model response.
        
        Args:
            prompt: The original prompt sent to the model
            response: The model's response
            task_type: The type of task (analysis, generation, editing, etc.)
            use_competitor: Whether to use a competing model for evaluation
            
        Returns:
            Dictionary with quality metrics and feedback
        """
        # If using a competitor model for evaluation
        if use_competitor and not hasattr(self, 'is_evaluator') and self.context.competitor_evaluation:
            competing_provider, competing_model = self._get_competing_model()
            
            if competing_provider and competing_model:
                # Interactive wait point before cross-model evaluation
                if hasattr(self.context, 'interactive') and self.context.interactive:
                    from src.core.interactive import interactive_wait
                    interactive_wait(
                        f"About to perform cross-model evaluation using {competing_provider.capitalize()} {competing_model}. " +
                        f"This will evaluate the quality of outputs from {self.context.provider.capitalize()} {self.context.model_name}.",
                        self.context.log_path,
                        self.context.interactive
                    )
                
                # Track the cost of this evaluation in the statistics
                self.context.process_statistics["evaluation_requests"] = self.context.process_statistics.get("evaluation_requests", 0) + 1
                
                try:
                    # Create an evaluation prompt
                    evaluation_prompt = f"""
                    You are an expert at evaluating AI model outputs for quality and correctness.
                    
                    ORIGINAL PROMPT:
                    ```
                    {prompt}
                    ```
                    
                    MODEL RESPONSE TO EVALUATE:
                    ```
                    {response}
                    ```
                    
                    TASK TYPE: {task_type}
                    
                    Please evaluate this response on a scale of 1-5 (where 5 is best) based on:
                    1. Relevance to the prompt
                    2. Accuracy and correctness
                    3. Completeness
                    4. Structure and clarity
                    5. Overall quality
                    
                    In your evaluation, identify:
                    - Specific issues with the response (if any)
                    - Highlights or strengths of the response (if any)
                    
                    IMPORTANT: Format your response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '{{' and end with '}}'.
                    
                    Example format:
                    {{
                      "quality_score": 4,
                      "quality_issues": ["Could be more concise", "Missing citation for key claim"],
                      "quality_highlights": ["Excellent analysis of results", "Well-structured arguments"],
                      "details": {{
                        "relevance": 5,
                        "accuracy": 4,
                        "completeness": 4,
                        "structure": 4
                      }}
                    }}
                    """
                    
                    # Create a temporary LLM client for the competing model
                    # Mark it as an evaluator to avoid infinite evaluation loops
                    evaluator_client = get_llm_client(competing_provider, competing_model, verify=False)
                    evaluator_client.is_evaluator = True  # Mark to prevent recursive evaluations
                    
                    # Get evaluation from the competing model
                    evaluation_response = evaluator_client.get_completion(
                        prompt=evaluation_prompt,
                        system_prompt="You are an expert evaluator of AI outputs. Evaluate fairly and objectively.",
                        max_tokens=1000
                    )
                    
                    # Track the evaluation cost
                    evaluation_tokens = evaluator_client.total_tokens_used
                    evaluation_cost = evaluator_client.total_cost
                    
                    self.context.process_statistics["evaluation_tokens"] = self.context.process_statistics.get("evaluation_tokens", 0) + evaluation_tokens
                    self.context.process_statistics["evaluation_cost"] = self.context.process_statistics.get("evaluation_cost", 0.0) + evaluation_cost
                    
                    # Parse the evaluation response
                    try:
                        # Extract JSON from the response using our helper method
                        from src.core.json_utils import extract_json_from_text
                        json_str = extract_json_from_text(evaluation_response)
                        evaluation = json.loads(json_str)
                        
                        # Add the evaluator info
                        evaluation["evaluator"] = f"{competing_provider}/{competing_model}"
                        evaluation["evaluation_cost"] = evaluation_cost
                        
                        # Log the cross-model evaluation
                        self.logger.info(f"Cross-model evaluation: {self.context.provider}/{self.context.model_name} evaluated by {competing_provider}/{competing_model}")
                        self.logger.info(f"Evaluation score: {evaluation.get('quality_score', 'N/A')}/5")
                        
                        # Interactive wait point after cross-model evaluation
                        if hasattr(self.context, 'interactive') and self.context.interactive:
                            from src.core.interactive import interactive_wait
                            score = evaluation.get('quality_score', 'N/A')
                            interactive_wait(
                                f"Cross-model evaluation completed with score: {score}/5. " +
                                f"The evaluation was performed by {competing_provider.capitalize()} {competing_model}.",
                                self.context.log_path,
                                self.context.interactive
                            )
                        
                        return evaluation
                    except Exception as parse_error:
                        self.logger.warning(f"Error parsing evaluation response: {parse_error}")
                        # Fall back to the basic evaluation
                
                except Exception as eval_error:
                    self.logger.warning(f"Error during competitor evaluation: {eval_error}")
                    # Fall back to the basic evaluation
        
        # Basic quality checks (used if competitor evaluation fails or is disabled)
        quality_score = 5  # Default high score
        quality_issues = []
        quality_highlights = []
        
        # Length check
        if len(response) < 50:
            quality_score -= 1
            quality_issues.append("Response is very short")
        elif len(response) > 10000 and task_type != "analysis":
            quality_score -= 1
            quality_issues.append("Response may be unnecessarily verbose")
        else:
            quality_highlights.append("Response length is appropriate")
        
        # Content relevance check
        relevance_score = 5
        prompt_keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        response_keywords = set(re.findall(r'\b\w{4,}\b', response.lower()))
        keyword_overlap = len(prompt_keywords.intersection(response_keywords)) / max(1, len(prompt_keywords))
        
        if keyword_overlap < 0.3:
            relevance_score -= 2
            quality_issues.append("Low relevance to prompt (few keyword matches)")
        elif keyword_overlap < 0.5:
            relevance_score -= 1
            quality_issues.append("Moderate relevance to prompt")
        else:
            quality_highlights.append("Good relevance to original prompt")
        
        # Structure check
        structure_score = 5
        if task_type == "analysis" and ":" not in response and len(response.split("\n")) < 3:
            structure_score -= 1
            quality_issues.append("Analysis lacks structure (no sections or bullet points)")
        
        # JSON validation if expected
        if "JSON" in prompt or "json" in prompt:
            try:
                json.loads(response.strip())
                quality_highlights.append("Valid JSON format")
            except (json.JSONDecodeError, ValueError) as e:
                quality_score -= 2
                quality_issues.append(f"Invalid JSON format: {e}")
        
        # Task-specific checks
        if task_type == "analysis":
            if not any(kw in response.lower() for kw in ["however", "but", "although", "while", "despite"]):
                quality_score -= 1
                quality_issues.append("Analysis lacks nuance or consideration of drawbacks")
            else:
                quality_highlights.append("Analysis shows nuanced thinking")
        
        # Completeness check
        if "list" in prompt.lower() and response.count("\n") < 3:
            quality_score -= 1
            quality_issues.append("Response may be incomplete (expected list items)")
        
        # Calculate final scores
        final_quality_score = min(5, max(1, int(round((quality_score + relevance_score + structure_score) / 3))))
        
        return {
            "quality_score": final_quality_score,
            "quality_issues": quality_issues,
            "quality_highlights": quality_highlights,
            "details": {
                "base_quality": quality_score,
                "relevance": relevance_score,
                "structure": structure_score,
                "keyword_overlap": keyword_overlap
            },
            "evaluator": "basic_algorithm"  # Mark this as an algorithmic evaluation
        }
    
    def _get_competing_model(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get a competing model for evaluation.
        
        Returns:
            Tuple of (provider, model_name) for the competing model
        """
        # Define competitor mapping by tier and provider
        competitors = {
            "anthropic": {
                "basic": {"openai": "gpt-4o-mini", "google": "gemini-1.5-flash"},
                "standard": {"openai": "gpt-4o", "google": "gemini-1.5-pro"},
                "advanced": {"openai": "gpt-4.5-preview", "google": "gemini-2.5-pro-preview"}
            },
            "openai": {
                "basic": {"anthropic": "claude-3-haiku-20240307", "google": "gemini-1.5-flash"},
                "standard": {"anthropic": "claude-3-5-sonnet-20241022", "google": "gemini-1.5-pro"},
                "advanced": {"anthropic": "claude-opus-4-20250514", "google": "gemini-2.5-pro-preview"}
            },
            "google": {
                "basic": {"anthropic": "claude-3-haiku-20240307", "openai": "gpt-4o-mini"},
                "standard": {"anthropic": "claude-3-5-sonnet-20241022", "openai": "gpt-4o"},
                "advanced": {"anthropic": "claude-opus-4-20250514", "openai": "gpt-4.5-preview"}
            }
        }
        
        # Determine current model's tier
        current_tier = "standard"  # Default tier
        
        # Look for tier indicators in model name
        if any(basic_indicator in self.context.model_name.lower() for basic_indicator in ["haiku", "mini", "flash"]):
            current_tier = "basic"
        elif any(advanced_indicator in self.context.model_name.lower() for advanced_indicator in ["opus", "4.5", "2.5"]):
            current_tier = "advanced"
        
        # Get competitors for this provider and tier
        tier_competitors = competitors.get(self.context.provider, {}).get(current_tier, {})
        
        # If we have a competing_evaluator specified, use that one
        if hasattr(self.context, 'competing_evaluator') and self.context.competing_evaluator:
            competing_provider, competing_model = self.context.competing_evaluator.split('/')
            if competing_provider != self.context.provider:  # Make sure it's actually a competitor
                return competing_provider, competing_model
        
        # Otherwise use the first available competitor
        for competing_provider, competing_model in tier_competitors.items():
            # Check if we have the API key for this provider
            env_var = f"{competing_provider.upper()}_API_KEY"
            if os.getenv(env_var):
                return competing_provider, competing_model
        
        # If no competitors available, return None/None
        return None, None