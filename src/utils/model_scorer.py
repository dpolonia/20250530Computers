"""Model scoring system based on provider specifications.

This module provides utilities for scoring and ranking LLM models
based on their specifications and capabilities.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import model information
from src.models.anthropic_models import CLAUDE_MODELS, get_claude_model_info
from src.models.openai_models import OPENAI_MODELS, get_openai_model_info
from src.models.google_models import GEMINI_MODELS, get_gemini_model_info

# Import database module for persistence
from src.utils.workflow_db import WorkflowDB

# Path to store the model scores as backup
MODEL_SCORES_PATH = "./.cache/model_scores.json"
os.makedirs(os.path.dirname(MODEL_SCORES_PATH), exist_ok=True)

logger = logging.getLogger(__name__)

def calculate_model_score(
    provider: str, 
    model_name: str, 
    model_info: Optional[Dict[str, Any]] = None,
    consider_cost: bool = False
) -> Tuple[int, float]:
    """Calculate a score for a model based on its specifications.
    
    Args:
        provider: The provider name ('anthropic', 'openai', 'google')
        model_name: The model name
        model_info: Optional model information dictionary
        consider_cost: Whether to factor cost into the scoring
        
    Returns:
        Tuple of (capability_score, cost_efficiency_score)
        capability_score: 0-100 indicating raw model capability
        cost_efficiency_score: 0-100 indicating cost efficiency
    """
    # Get model info if not provided
    if not model_info:
        if provider == "anthropic":
            model_info = get_claude_model_info(model_name)
        elif provider == "openai":
            model_info = get_openai_model_info(model_name)
        elif provider == "google":
            model_info = get_gemini_model_info(model_name)
            
    if not model_info:
        logger.warning(f"No model info found for {provider}/{model_name}")
        return 50, 50  # Default middle scores
    
    # Base capability score starts at 50
    capability_score = 50
    
    # Score based on context size
    max_tokens = model_info.get("max_tokens", 0)
    if max_tokens >= 1000000:  # 1M+ context
        capability_score += 25
    elif max_tokens >= 128000:  # 128K context
        capability_score += 20
    elif max_tokens >= 32000:  # 32K context
        capability_score += 15
    elif max_tokens >= 16000:  # 16K context
        capability_score += 10
    elif max_tokens >= 8000:  # 8K context
        capability_score += 5
    
    # Score based on description (power level indicated by provider)
    description = model_info.get("description", "").lower()
    if "most powerful" in description:
        capability_score += 20
    elif "powerful" in description:
        capability_score += 15
    elif "balanced" in description:
        capability_score += 10
    elif "fast" in description:
        capability_score += 5
    
    # Score based on model generation/version
    if provider == "anthropic":
        if "opus" in model_name.lower() or "opus-4" in model_name.lower():
            capability_score += 20
        elif "sonnet" in model_name.lower():
            capability_score += 10
        elif "haiku" in model_name.lower():
            capability_score += 5
            
    elif provider == "openai":
        if "4.5" in model_name.lower() or "o1" in model_name.lower():
            capability_score += 20
        elif "4o" in model_name.lower() and "mini" not in model_name.lower():
            capability_score += 15
        elif "4" in model_name.lower() and "mini" not in model_name.lower():
            capability_score += 10
        elif "mini" in model_name.lower():
            capability_score += 5
            
    elif provider == "google":
        if "2.5-pro" in model_name.lower():
            capability_score += 20
        elif "2.0-pro" in model_name.lower() or "1.5-pro" in model_name.lower():
            capability_score += 15
        elif "flash" in model_name.lower():
            capability_score += 10
    
    # Ensure capability score is within 0-100 range
    capability_score = max(0, min(100, capability_score))
    
    # Calculate cost efficiency score
    cost_efficiency_score = 50  # Start at neutral
    
    # Get pricing information
    input_cost = model_info.get("price_per_1k_input", 0)
    output_cost = model_info.get("price_per_1k_output", 0)
    avg_cost = (input_cost + output_cost) / 2
    
    # Scale is logarithmic to better differentiate the models
    # Most expensive (opus-like): ~0.045 avg → score ~20
    # Medium (4o-like): ~0.01 avg → score ~50
    # Cheapest (haiku/flash-like): ~0.0003 avg → score ~90
    if avg_cost <= 0.0001:  # Extremely cheap
        cost_efficiency_score = 95
    elif avg_cost <= 0.0005:  # Very cheap
        cost_efficiency_score = 90
    elif avg_cost <= 0.001:  # Cheap
        cost_efficiency_score = 80
    elif avg_cost <= 0.002:  # Moderately cheap
        cost_efficiency_score = 70
    elif avg_cost <= 0.005:  # Medium
        cost_efficiency_score = 60
    elif avg_cost <= 0.01:  # Moderately expensive
        cost_efficiency_score = 50
    elif avg_cost <= 0.02:  # Expensive
        cost_efficiency_score = 40
    elif avg_cost <= 0.04:  # Very expensive
        cost_efficiency_score = 30
    else:  # Extremely expensive
        cost_efficiency_score = 20
    
    return capability_score, cost_efficiency_score

def initialize_model_scores(force_update: bool = False) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Initialize scores for all models from all providers.
    
    Args:
        force_update: Whether to force an update of all model scores
        
    Returns:
        Dictionary mapping provider/model to score details
    """
    # Create database connection
    db = WorkflowDB()
    
    # Check if update is due
    update_due = force_update or db.is_model_update_due()
    
    # Try to get scores from database first (if no update is due)
    if not update_due:
        db_scores = db.get_model_scores_from_db()
        if db_scores and all(provider in db_scores for provider in ["anthropic", "openai", "google"]):
            logger.info("Using model scores from database")
            db.close()
            
            # Save scores to file as backup
            try:
                with open(MODEL_SCORES_PATH, 'w') as f:
                    json.dump(db_scores, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save model scores to file: {e}")
                
            return db_scores
    
    # If update is due or no scores in database, recalculate
    logger.info("Updating model scores - biweekly check is due" if update_due else "Initializing model scores")
    
    scores = {}
    
    # Score Anthropic models
    scores["anthropic"] = {}
    for model_name, model_info in CLAUDE_MODELS.items():
        capability, cost_efficiency = calculate_model_score("anthropic", model_name, model_info)
        scores["anthropic"][model_name] = {
            "capability": capability,
            "cost_efficiency": cost_efficiency,
            "input_cost": model_info.get("price_per_1k_input", 0),
            "output_cost": model_info.get("price_per_1k_output", 0),
            "max_tokens": model_info.get("max_tokens", 0),
            "description": model_info.get("description", "")
        }
        
        # Store in database
        db.store_model_score(
            "anthropic", 
            model_name, 
            capability, 
            cost_efficiency,
            model_info.get("price_per_1k_input", 0),
            model_info.get("price_per_1k_output", 0),
            model_info.get("max_tokens", 0),
            model_info.get("description", "")
        )
    
    # Score OpenAI models
    scores["openai"] = {}
    for model_name, model_info in OPENAI_MODELS.items():
        capability, cost_efficiency = calculate_model_score("openai", model_name, model_info)
        scores["openai"][model_name] = {
            "capability": capability,
            "cost_efficiency": cost_efficiency,
            "input_cost": model_info.get("price_per_1k_input", 0),
            "output_cost": model_info.get("price_per_1k_output", 0),
            "max_tokens": model_info.get("max_tokens", 0),
            "description": model_info.get("description", "")
        }
        
        # Store in database
        db.store_model_score(
            "openai", 
            model_name, 
            capability, 
            cost_efficiency,
            model_info.get("price_per_1k_input", 0),
            model_info.get("price_per_1k_output", 0),
            model_info.get("max_tokens", 0),
            model_info.get("description", "")
        )
    
    # Score Google models
    scores["google"] = {}
    for model_name, model_info in GEMINI_MODELS.items():
        capability, cost_efficiency = calculate_model_score("google", model_name, model_info)
        scores["google"][model_name] = {
            "capability": capability,
            "cost_efficiency": cost_efficiency,
            "input_cost": model_info.get("price_per_1k_input", 0),
            "output_cost": model_info.get("price_per_1k_output", 0),
            "max_tokens": model_info.get("max_tokens", 0),
            "description": model_info.get("description", "")
        }
        
        # Store in database
        db.store_model_score(
            "google", 
            model_name, 
            capability, 
            cost_efficiency,
            model_info.get("price_per_1k_input", 0),
            model_info.get("price_per_1k_output", 0),
            model_info.get("max_tokens", 0),
            model_info.get("description", "")
        )
    
    # Update the check schedule
    db.update_model_check_schedule("completed")
    
    # Close database connection
    db.close()
    
    # Save scores to file as backup
    try:
        with open(MODEL_SCORES_PATH, 'w') as f:
            json.dump(scores, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save model scores to file: {e}")
    
    return scores

def get_model_scores() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Get model scores from database or initialize if not available.
    
    Returns:
        Dictionary mapping provider/model to score details
    """
    # Create database connection
    db = WorkflowDB()
    
    # Try to get scores from database
    db_scores = db.get_model_scores_from_db()
    
    # If we have scores in the database, check if they're complete
    if db_scores:
        update_needed = False
        
        # Check if all providers are present
        if not all(provider in db_scores for provider in ["anthropic", "openai", "google"]):
            update_needed = True
        
        # Check Anthropic models
        if not update_needed:
            for model_name in CLAUDE_MODELS:
                if "anthropic" not in db_scores or model_name not in db_scores["anthropic"]:
                    update_needed = True
                    break
                    
        # Check OpenAI models
        if not update_needed:
            for model_name in OPENAI_MODELS:
                if "openai" not in db_scores or model_name not in db_scores["openai"]:
                    update_needed = True
                    break
                    
        # Check Google models
        if not update_needed:
            for model_name in GEMINI_MODELS:
                if "google" not in db_scores or model_name not in db_scores["google"]:
                    update_needed = True
                    break
        
        # Also check if it's time for the biweekly update
        if not update_needed:
            update_needed = db.is_model_update_due()
        
        # Close database connection
        db.close()
        
        if update_needed:
            logger.info("Model update needed. Reinitializing scores.")
            return initialize_model_scores(force_update=True)
        
        return db_scores
    
    # No scores in database, initialize
    logger.info("No model scores found in database. Initializing.")
    return initialize_model_scores()

def score_model(provider: str, model_name: str) -> Dict[str, Any]:
    """Get the scores for a specific model.
    
    Args:
        provider: The provider name ('anthropic', 'openai', 'google')
        model_name: The model name
        
    Returns:
        Dictionary with capability and cost_efficiency scores
    """
    scores = get_model_scores()
    
    # Remove description if present
    base_model = model_name.split(" (")[0]
    
    if provider in scores and base_model in scores[provider]:
        return scores[provider][base_model]
    
    # If not found in cache, calculate on the fly
    capability, cost_efficiency = calculate_model_score(provider, base_model)
    return {
        "capability": capability,
        "cost_efficiency": cost_efficiency,
        "input_cost": 0,  # Unknown
        "output_cost": 0  # Unknown
    }

def get_best_models_by_mode(
    operation_mode: str, 
    provider_runs: Dict[str, Dict],
    prioritize_cost: bool = False
) -> List[Dict[str, Any]]:
    """Rank the models from provider runs by their capability scores.
    
    Args:
        operation_mode: The operation mode ('training', 'finetuning', 'final')
        provider_runs: Dictionary mapping providers to their run information
        prioritize_cost: Whether to prioritize cost efficiency over capability
        
    Returns:
        List of models sorted by score (highest first)
    """
    model_scores = []
    
    # Get the model scores
    scores = get_model_scores()
    
    # Set weight for blending capability vs cost-efficiency based on operation mode
    # Training mode → heavily favor cost
    # Fine-tuning mode → balance capability and cost
    # Final mode → heavily favor capability
    cost_weight = 0.8 if operation_mode == "training" else (
                 0.5 if operation_mode == "finetuning" else 0.2)
    
    # Override with prioritize_cost if set
    if prioritize_cost:
        cost_weight = 0.8
    
    # Score each model in provider_runs
    for provider_name, run in provider_runs.items():
        model_name = run.get('model', '')
        base_model = model_name.split(" (")[0]
        
        # Get scores from cache or calculate
        if provider_name in scores and base_model in scores[provider_name]:
            model_scores_dict = scores[provider_name][base_model]
            capability = model_scores_dict.get("capability", 50)
            cost_efficiency = model_scores_dict.get("cost_efficiency", 50)
            input_cost = model_scores_dict.get("input_cost", 0)
            output_cost = model_scores_dict.get("output_cost", 0)
        else:
            capability, cost_efficiency = calculate_model_score(provider_name, base_model)
            input_cost = 0
            output_cost = 0
        
        # Calculate blended score
        blended_score = (capability * (1 - cost_weight)) + (cost_efficiency * cost_weight)
        
        model_scores.append({
            "provider": provider_name,
            "model": model_name,
            "capability": capability,
            "cost_efficiency": cost_efficiency,
            "blended_score": blended_score,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "run_id": run.get('run_id')
        })
    
    # Sort by blended score (highest first)
    model_scores.sort(key=lambda x: x["blended_score"], reverse=True)
    
    return model_scores