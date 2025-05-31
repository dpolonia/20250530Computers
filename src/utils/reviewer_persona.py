"""Reviewer Persona Generation and Simulation

This module provides functionality to create and simulate academic reviewer personas
based on initial reviewer comments, journal guidelines, and field expertise.
It integrates with the FinePersonas dataset from Hugging Face for enhanced persona generation.
"""

import os
import json
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import functools
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache directory for HuggingFace datasets
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Fallback reviewer persona archetypes if HuggingFace is not available
REVIEWER_ARCHETYPES = {
    "methodologist": {
        "focus": ["methodology", "experimental design", "statistical analysis", "validity"],
        "tone": "analytical",
        "depth": "detailed",
        "expertise": "research methods",
        "bias": "favors methodological rigor over novelty",
        "strictness": 0.8
    },
    "domain_expert": {
        "focus": ["accuracy", "contribution", "literature", "state-of-the-art"],
        "tone": "authoritative",
        "depth": "comprehensive",
        "expertise": "domain knowledge",
        "bias": "expects significant contribution to the field",
        "strictness": 0.7
    },
    "clarity_advocate": {
        "focus": ["clarity", "structure", "readability", "presentation"],
        "tone": "constructive",
        "depth": "moderate",
        "expertise": "academic writing",
        "bias": "emphasizes clear communication over technical complexity",
        "strictness": 0.6
    },
    "novelty_seeker": {
        "focus": ["novelty", "innovation", "originality", "future work"],
        "tone": "enthusiastic",
        "depth": "conceptual",
        "expertise": "emerging trends",
        "bias": "favors novelty over incremental improvements",
        "strictness": 0.5
    },
    "meticulous_editor": {
        "focus": ["details", "formatting", "grammar", "references"],
        "tone": "precise",
        "depth": "thorough",
        "expertise": "publishing standards",
        "bias": "highly attentive to publication standards",
        "strictness": 0.9
    },
    "practical_applicator": {
        "focus": ["application", "practical implications", "industry relevance", "impact"],
        "tone": "pragmatic",
        "depth": "application-focused",
        "expertise": "industry applications",
        "bias": "values practical utility over theoretical contribution",
        "strictness": 0.6
    },
    "theoretical_purist": {
        "focus": ["theory", "conceptual framework", "underlying assumptions", "theoretical contribution"],
        "tone": "scholarly",
        "depth": "theoretical",
        "expertise": "foundational theories",
        "bias": "emphasizes theoretical soundness",
        "strictness": 0.8
    },
    "cross_disciplinary_connector": {
        "focus": ["interdisciplinary connections", "broader impact", "cross-field relevance"],
        "tone": "exploratory",
        "depth": "connective",
        "expertise": "multiple disciplines",
        "bias": "values integration of knowledge across fields",
        "strictness": 0.5
    }
}

# FinePersona cache with TTL
FINE_PERSONAS_CACHE = {}
FINE_PERSONAS_CACHE_TTL = 3600  # 1 hour in seconds

def analyze_reviewer_comments(comments: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze reviewer comments to extract reviewer characteristics.
    
    Args:
        comments: Dictionary containing reviewer comments
        
    Returns:
        Dictionary with extracted reviewer characteristics
    """
    characteristics = {
        "focus_areas": [],
        "tone": "",
        "strictness": 0.0,
        "expertise": [],
        "common_criticisms": []
    }
    
    # Extract focus areas from main concerns and required changes
    if "main_concerns" in comments:
        for concern in comments["main_concerns"]:
            # Extract key terms from concern
            terms = _extract_key_terms(concern)
            if terms:
                characteristics["focus_areas"].extend(terms)
    
    if "required_changes" in comments:
        for change in comments["required_changes"]:
            terms = _extract_key_terms(change)
            if terms:
                characteristics["focus_areas"].extend(terms)
                characteristics["common_criticisms"].append(change)
    
    # Estimate strictness based on overall assessment and language
    if "overall_assessment" in comments:
        assessment = comments["overall_assessment"].lower()
        if "reject" in assessment:
            characteristics["strictness"] = 0.9
        elif "major revision" in assessment:
            characteristics["strictness"] = 0.7
        elif "minor revision" in assessment:
            characteristics["strictness"] = 0.5
        elif "accept" in assessment:
            characteristics["strictness"] = 0.3
    
    # Remove duplicates and limit to top 5
    characteristics["focus_areas"] = list(set(characteristics["focus_areas"]))[:5]
    characteristics["common_criticisms"] = list(set(characteristics["common_criticisms"]))[:3]
    
    return characteristics

def _extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text.
    
    Args:
        text: Input text
        
    Returns:
        List of key terms
    """
    # List of key terms to look for in reviewer comments
    key_terms = {
        "methodology": ["method", "approach", "procedure", "technique", "protocol"],
        "clarity": ["clear", "unclear", "ambiguous", "vague", "readability", "understandable"],
        "novelty": ["novel", "innovative", "original", "new", "unique", "contribution"],
        "literature": ["literature", "citation", "reference", "prior work", "previous work"],
        "structure": ["structure", "organization", "flow", "section", "paragraph"],
        "validity": ["valid", "invalid", "validity", "reliable", "reliability", "reproducible"],
        "statistics": ["statistics", "statistical", "analysis", "significance", "p-value", "test"],
        "theoretical": ["theory", "theoretical", "framework", "model", "concept"],
        "practical": ["practical", "application", "applied", "real-world", "industry"],
        "writing": ["writing", "grammar", "spelling", "language", "style", "tone"],
        "formatting": ["format", "formatting", "figure", "table", "equation", "layout"]
    }
    
    # Check for each key term
    found_terms = []
    for category, terms in key_terms.items():
        for term in terms:
            if term.lower() in text.lower():
                found_terms.append(category)
                break
    
    return found_terms

@functools.lru_cache(maxsize=128)
def load_fine_personas_dataset(subset_size: int = 1000) -> List[Dict[str, Any]]:
    """Load and preprocess the FinePersonas dataset.
    
    Args:
        subset_size: Number of personas to load (defaults to 1000 for memory efficiency)
        
    Returns:
        List of preprocessed personas from the dataset
    """
    if not HUGGINGFACE_AVAILABLE:
        logger.warning("HuggingFace libraries not available. Using fallback archetypes.")
        return []
    
    try:
        # Check cache first
        cache_key = f"fine_personas_{subset_size}"
        if cache_key in FINE_PERSONAS_CACHE:
            cache_entry = FINE_PERSONAS_CACHE[cache_key]
            # Check if cache is still valid
            if time.time() - cache_entry["timestamp"] < FINE_PERSONAS_CACHE_TTL:
                logger.info(f"Using cached FinePersonas dataset ({len(cache_entry['data'])} entries)")
                return cache_entry["data"]
        
        # Load the dataset from Hugging Face
        logger.info("Loading FinePersonas dataset from Hugging Face...")
        dataset = load_dataset(
            "argilla/FinePersonas-v0.1",
            split="train",
            cache_dir=HF_CACHE_DIR
        )
        
        # Filter for academic and scientific personas
        academic_keywords = [
            "professor", "researcher", "scientist", "academic", "scholar", "lecturer", 
            "phd", "doctoral", "university", "research", "faculty", "journal", "review"
        ]
        
        academic_personas = []
        
        # Process and filter the dataset
        for idx, entry in enumerate(dataset):
            if idx >= subset_size * 10:  # Check more entries than we need to ensure we get enough matches
                break
                
            description = entry.get("persona", "").lower()
            
            # Check if this is an academic persona
            if any(keyword in description for keyword in academic_keywords):
                # Extract relevant information
                persona = {
                    "id": entry.get("id", str(idx)),
                    "description": entry.get("persona", ""),
                    "focus_areas": [],
                    "expertise": [],
                    "tone": "",
                    "strictness": random.uniform(0.4, 0.9),  # Random initial strictness
                    "depth": "",
                    "bias": ""
                }
                
                # Extract focus areas and expertise from description
                if "expertise in" in description or "specialized in" in description:
                    expertise_parts = description.split("expertise in")[1].split(".")[0] if "expertise in" in description else description.split("specialized in")[1].split(".")[0]
                    expertise_keywords = [kw.strip() for kw in expertise_parts.split(",")]
                    persona["expertise"] = expertise_keywords[:3]  # Limit to top 3
                
                # Determine tone and depth from description
                if "critical" in description or "rigorous" in description:
                    persona["tone"] = "analytical"
                    persona["depth"] = "detailed"
                    persona["strictness"] = random.uniform(0.7, 0.9)
                elif "constructive" in description or "helpful" in description:
                    persona["tone"] = "constructive"
                    persona["depth"] = "moderate"
                    persona["strictness"] = random.uniform(0.4, 0.6)
                elif "authoritative" in description or "expert" in description:
                    persona["tone"] = "authoritative"
                    persona["depth"] = "comprehensive"
                    persona["strictness"] = random.uniform(0.6, 0.8)
                else:
                    # Assign random tone and depth if not found
                    persona["tone"] = random.choice(["analytical", "constructive", "authoritative", "scholarly"])
                    persona["depth"] = random.choice(["detailed", "moderate", "comprehensive", "thorough"])
                
                # Determine focus areas
                for key, focus_terms in {
                    "methodology": ["method", "approach", "protocol", "experiment"],
                    "clarity": ["clear", "writing", "communication", "presentation"],
                    "theory": ["theory", "framework", "concept", "principle"],
                    "practical": ["application", "practice", "implementation", "industry"],
                    "details": ["detail", "careful", "meticulous", "thorough"],
                    "novelty": ["novel", "innovative", "original", "creative"]
                }.items():
                    if any(term in description for term in focus_terms):
                        persona["focus_areas"].append(key)
                
                # If no focus areas were identified, assign general ones
                if not persona["focus_areas"]:
                    persona["focus_areas"] = ["research", "academic writing"]
                
                # Add bias based on focus areas
                if "methodology" in persona["focus_areas"]:
                    persona["bias"] = "favors methodological rigor over novelty"
                elif "novelty" in persona["focus_areas"]:
                    persona["bias"] = "favors innovation over incremental improvements"
                elif "practical" in persona["focus_areas"]:
                    persona["bias"] = "values practical utility over theoretical contribution"
                elif "theory" in persona["focus_areas"]:
                    persona["bias"] = "emphasizes theoretical soundness"
                else:
                    persona["bias"] = "balanced approach to review"
                
                academic_personas.append(persona)
                
                # If we have enough personas, stop processing
                if len(academic_personas) >= subset_size:
                    break
        
        # Cache the results
        FINE_PERSONAS_CACHE[cache_key] = {
            "data": academic_personas,
            "timestamp": time.time()
        }
        
        logger.info(f"Loaded {len(academic_personas)} academic personas from FinePersonas dataset")
        return academic_personas
    
    except Exception as e:
        logger.error(f"Error loading FinePersonas dataset: {e}")
        return []

def match_reviewer_to_archetype(characteristics: Dict[str, Any]) -> str:
    """Match reviewer characteristics to the closest archetype.
    
    Args:
        characteristics: Dictionary with reviewer characteristics
        
    Returns:
        Name of the closest matching archetype
    """
    # First try to match with FinePersonas if available
    if HUGGINGFACE_AVAILABLE:
        try:
            fine_personas = load_fine_personas_dataset()
            if fine_personas:
                best_persona_match = None
                best_persona_score = -1
                
                for persona in fine_personas:
                    score = 0
                    
                    # Match focus areas
                    for focus_area in characteristics["focus_areas"]:
                        if focus_area in persona["focus_areas"] or any(f in focus_area for f in persona["focus_areas"]):
                            score += 1
                    
                    # Match strictness
                    if "strictness" in persona:
                        strictness_diff = abs(characteristics["strictness"] - persona["strictness"])
                        score += (1 - strictness_diff)
                    
                    if score > best_persona_score:
                        best_persona_score = score
                        best_persona_match = persona
                
                if best_persona_match and best_persona_score > 0:
                    # Create a custom archetype ID based on the persona ID
                    archetype_id = f"fine_persona_{best_persona_match['id']}"
                    
                    # Add to archetypes dictionary for later use
                    REVIEWER_ARCHETYPES[archetype_id] = {
                        "focus": best_persona_match["focus_areas"],
                        "tone": best_persona_match["tone"],
                        "depth": best_persona_match["depth"],
                        "expertise": best_persona_match.get("expertise", ["academic research"]),
                        "bias": best_persona_match["bias"],
                        "strictness": best_persona_match["strictness"],
                        "description": best_persona_match["description"]
                    }
                    
                    logger.info(f"Matched reviewer to FinePersona: {archetype_id}")
                    return archetype_id
        except Exception as e:
            logger.warning(f"Error matching with FinePersonas, falling back to standard archetypes: {e}")
    
    # Fall back to standard archetype matching
    best_match = None
    best_score = -1
    
    for archetype_name, archetype in REVIEWER_ARCHETYPES.items():
        # Skip dynamic FinePersona archetypes for the fallback matching
        if archetype_name.startswith("fine_persona_"):
            continue
            
        score = 0
        
        # Match focus areas
        for focus_area in characteristics["focus_areas"]:
            if focus_area in archetype["focus"] or any(f in focus_area for f in archetype["focus"]):
                score += 1
        
        # Match strictness
        strictness_diff = abs(characteristics["strictness"] - archetype["strictness"])
        score += (1 - strictness_diff)
        
        if score > best_score:
            best_score = score
            best_match = archetype_name
    
    return best_match or "domain_expert"  # Default to domain expert if no good match

def generate_reviewer_persona(archetype: str, characteristics: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Generate a detailed reviewer persona based on archetype and characteristics.
    
    Args:
        archetype: Name of the reviewer archetype
        characteristics: Dictionary with reviewer characteristics
        field: Academic field
        
    Returns:
        Dictionary with detailed reviewer persona
    """
    base_archetype = REVIEWER_ARCHETYPES[archetype]
    
    # Check if this is a FinePersona-based archetype
    if archetype.startswith("fine_persona_") and "description" in base_archetype:
        # For FinePersona-based archetypes, we have richer information
        description = base_archetype.get("description", "")
        
        # Create persona with FinePersona details
        persona = {
            "archetype": archetype,
            "field": field,
            "focus_areas": base_archetype["focus"] + characteristics.get("focus_areas", []),
            "tone": base_archetype["tone"],
            "depth": base_archetype["depth"],
            "expertise": base_archetype.get("expertise", ["academic research"]) + characteristics.get("expertise", []),
            "strictness": (base_archetype["strictness"] + characteristics.get("strictness", 0.5)) / 2,
            "bias": base_archetype["bias"],
            "common_criticisms": characteristics.get("common_criticisms", []),
            "is_fine_persona": True,
            "fine_persona_description": description
        }
        
        # Extract background from the FinePersona description
        # The FinePersona descriptions are often detailed enough to serve as background
        background_parts = []
        
        # Extract key sentences that describe the persona's background
        for sentence in description.split(". "):
            if any(keyword in sentence.lower() for keyword in ["professor", "researcher", "scientist", "academic", "background", "experience", "specialized", "expert"]):
                background_parts.append(sentence)
        
        if background_parts:
            persona["background"] = ". ".join(background_parts[:3]) + "."  # Limit to first 3 relevant sentences
        else:
            # Fall back to generating background if nothing relevant was found
            years_experience = random.randint(5, 25)
            institution_types = ["research university", "teaching-focused university", "industry research lab", 
                               "government research institution", "prestigious private university"]
            institution_type = random.choice(institution_types)
            
            persona["background"] = f"A {persona['field']} researcher with {years_experience} years of experience at a {institution_type}. "
            
            if persona["strictness"] > 0.7:
                persona["background"] += "Known for maintaining high standards in peer review. "
            elif persona["strictness"] < 0.4:
                persona["background"] += "Generally supportive and encouraging in peer review. "
            
            expertise_str = ", ".join([exp for exp in persona["expertise"][:2] if isinstance(exp, str)])
            if expertise_str:
                persona["background"] += f"Specialized in {expertise_str}."
        
        # Generate a review style based on the FinePersona characteristics
        persona["review_style"] = f"Reviews in a {persona['tone']} tone with {persona['depth']} depth. "
        
        focus_str = ", ".join(persona["focus_areas"][:3])
        persona["review_style"] += f"Pays particular attention to {focus_str}. "
        
        if persona["strictness"] > 0.8:
            persona["review_style"] += "Extremely thorough and demanding in assessments."
        elif persona["strictness"] > 0.6:
            persona["review_style"] += "Fairly rigorous in evaluations."
        elif persona["strictness"] > 0.4:
            persona["review_style"] += "Balanced in criticism and praise."
        else:
            persona["review_style"] += "Generally lenient, focusing on positive aspects."
        
        return persona
    
    # For traditional archetypes, use the original logic
    persona = {
        "archetype": archetype,
        "field": field,
        "focus_areas": base_archetype["focus"] + characteristics.get("focus_areas", []),
        "tone": base_archetype["tone"],
        "depth": base_archetype["depth"],
        "expertise": [base_archetype["expertise"]] + characteristics.get("expertise", []),
        "strictness": (base_archetype["strictness"] + characteristics.get("strictness", 0.5)) / 2,
        "bias": base_archetype["bias"],
        "common_criticisms": characteristics.get("common_criticisms", []),
        "is_fine_persona": False
    }
    
    # Generate a background story
    years_experience = random.randint(5, 25)
    institution_types = ["research university", "teaching-focused university", "industry research lab", 
                         "government research institution", "prestigious private university"]
    institution_type = random.choice(institution_types)
    
    persona["background"] = f"A {persona['field']} researcher with {years_experience} years of experience at a {institution_type}. "
    
    if persona["strictness"] > 0.7:
        persona["background"] += "Known for maintaining high standards in peer review. "
    elif persona["strictness"] < 0.4:
        persona["background"] += "Generally supportive and encouraging in peer review. "
    
    expertise_str = ", ".join([exp for exp in persona["expertise"][:2] if isinstance(exp, str)])
    persona["background"] += f"Specialized in {expertise_str}."
    
    # Generate a review style
    persona["review_style"] = f"Reviews in a {persona['tone']} tone with {persona['depth']} depth. "
    
    focus_str = ", ".join(persona["focus_areas"][:3])
    persona["review_style"] += f"Pays particular attention to {focus_str}. "
    
    if persona["strictness"] > 0.8:
        persona["review_style"] += "Extremely thorough and demanding in assessments."
    elif persona["strictness"] > 0.6:
        persona["review_style"] += "Fairly rigorous in evaluations."
    elif persona["strictness"] > 0.4:
        persona["review_style"] += "Balanced in criticism and praise."
    else:
        persona["review_style"] += "Generally lenient, focusing on positive aspects."
    
    return persona

def create_reviewer_personas(reviewer_comments: List[Dict[str, Any]], journal_guidelines: str, field: str, 
                           personas_per_reviewer: int = 4) -> List[Dict[str, Any]]:
    """Create a set of reviewer personas based on original reviewer comments.
    
    Args:
        reviewer_comments: List of dictionaries containing reviewer comments
        journal_guidelines: Text of journal review guidelines
        field: Academic field of the paper
        personas_per_reviewer: Number of personas to create per reviewer (default: 4)
        
    Returns:
        List of reviewer personas grouped by original reviewer
    """
    all_personas = []
    used_archetypes = set()
    
    # Process each reviewer's comments
    for i, comments in enumerate(reviewer_comments):
        reviewer_personas = []
        
        # Analyze reviewer comments
        characteristics = analyze_reviewer_comments(comments)
        
        # Create multiple personas for this reviewer
        for persona_index in range(personas_per_reviewer):
            # For first persona, use direct matching
            if persona_index == 0:
                # Match to closest archetype
                matched_archetype = match_reviewer_to_archetype(characteristics)
                
                # Ensure we don't use the same archetype twice across all reviewers
                while matched_archetype in used_archetypes and len(used_archetypes) < len(REVIEWER_ARCHETYPES):
                    # Get the next best archetype
                    alternatives = set(REVIEWER_ARCHETYPES.keys()) - used_archetypes
                    matched_archetype = random.choice(list(alternatives))
                
                used_archetypes.add(matched_archetype)
                
                # Generate detailed persona - primary persona most closely matches the original reviewer
                persona = generate_reviewer_persona(matched_archetype, characteristics, field)
                persona["is_primary"] = True
                
            else:
                # For additional personas, create varied perspectives
                # Modify characteristics slightly for diversity
                varied_characteristics = characteristics.copy()
                
                # Vary focus areas by adding/removing some
                if "focus_areas" in varied_characteristics and varied_characteristics["focus_areas"]:
                    # Remove 1-2 focus areas if there are enough
                    if len(varied_characteristics["focus_areas"]) > 2:
                        to_remove = random.randint(1, min(2, len(varied_characteristics["focus_areas"])-1))
                        for _ in range(to_remove):
                            varied_characteristics["focus_areas"].pop(random.randrange(len(varied_characteristics["focus_areas"])))
                    
                    # Add 1-2 random focus areas
                    possible_focus_areas = ["methodology", "clarity", "novelty", "literature", 
                                          "structure", "validity", "statistics", "theoretical", 
                                          "practical", "writing", "formatting"]
                    added_areas = random.sample([a for a in possible_focus_areas 
                                               if a not in varied_characteristics["focus_areas"]], 
                                              k=min(2, len(possible_focus_areas)))
                    varied_characteristics["focus_areas"].extend(added_areas)
                
                # Vary strictness by ±0.2
                if "strictness" in varied_characteristics:
                    strictness_variation = random.uniform(-0.2, 0.2)
                    varied_characteristics["strictness"] = max(0.1, min(0.9, 
                                                                     varied_characteristics["strictness"] + strictness_variation))
                
                # Use a different matching approach for additional personas
                if persona_index == 1 and HUGGINGFACE_AVAILABLE:
                    # Second persona should preferably be a FinePersona if available
                    try:
                        fine_personas = load_fine_personas_dataset()
                        if fine_personas:
                            # Select a random FinePersona
                            fine_persona = random.choice(fine_personas)
                            fine_archetype_id = f"fine_persona_{fine_persona['id']}"
                            
                            # Add to archetypes dictionary for later use
                            REVIEWER_ARCHETYPES[fine_archetype_id] = {
                                "focus": fine_persona["focus_areas"],
                                "tone": fine_persona["tone"],
                                "depth": fine_persona["depth"],
                                "expertise": fine_persona.get("expertise", ["academic research"]),
                                "bias": fine_persona["bias"],
                                "strictness": fine_persona["strictness"],
                                "description": fine_persona["description"]
                            }
                            
                            matched_archetype = fine_archetype_id
                        else:
                            # Fall back to random unused archetype
                            unused_archetypes = set(REVIEWER_ARCHETYPES.keys()) - used_archetypes
                            if not unused_archetypes:
                                unused_archetypes = set(REVIEWER_ARCHETYPES.keys())
                            
                            matched_archetype = random.choice(list(unused_archetypes))
                    except Exception as e:
                        logger.warning(f"Error using FinePersona for second persona: {e}")
                        # Fall back to random unused archetype
                        unused_archetypes = set(REVIEWER_ARCHETYPES.keys()) - used_archetypes
                        if not unused_archetypes:
                            unused_archetypes = set(REVIEWER_ARCHETYPES.keys())
                        
                        matched_archetype = random.choice(list(unused_archetypes))
                else:
                    # For other additional personas, use random archetypes
                    unused_archetypes = set(REVIEWER_ARCHETYPES.keys()) - used_archetypes
                    if not unused_archetypes:
                        unused_archetypes = set(REVIEWER_ARCHETYPES.keys())
                    
                    matched_archetype = random.choice(list(unused_archetypes))
                
                used_archetypes.add(matched_archetype)
                
                # Generate persona with varied characteristics
                persona = generate_reviewer_persona(matched_archetype, varied_characteristics, field)
                persona["is_primary"] = False
            
            # Add common attributes
            persona["original_reviewer_number"] = i + 1
            persona["persona_index"] = persona_index + 1
            
            reviewer_personas.append(persona)
        
        # Add this reviewer's personas to the overall list
        all_personas.append({
            "reviewer_id": i + 1,
            "personas": reviewer_personas
        })
    
    # If we need more reviewers to reach 3
    while len(all_personas) < 3:
        reviewer_id = len(all_personas) + 1
        reviewer_personas = []
        
        # Create personas for additional reviewer
        for persona_index in range(personas_per_reviewer):
            # Select unused archetypes
            unused_archetypes = set(REVIEWER_ARCHETYPES.keys()) - used_archetypes
            if not unused_archetypes:
                unused_archetypes = set(REVIEWER_ARCHETYPES.keys())
            
            archetype = random.choice(list(unused_archetypes))
            used_archetypes.add(archetype)
            
            # Create generic characteristics
            characteristics = {
                "focus_areas": [],
                "strictness": random.uniform(0.4, 0.8),
                "expertise": []
            }
            
            # Generate detailed persona
            persona = generate_reviewer_persona(archetype, characteristics, field)
            persona["original_reviewer_number"] = None  # Not based on an original reviewer
            persona["persona_index"] = persona_index + 1
            persona["is_primary"] = (persona_index == 0)  # First persona is primary
            
            reviewer_personas.append(persona)
        
        all_personas.append({
            "reviewer_id": reviewer_id,
            "personas": reviewer_personas
        })
    
    return all_personas

def generate_review_instructions(persona: Dict[str, Any], journal_guidelines: str) -> str:
    """Generate instructions for creating a review based on persona and journal guidelines.
    
    Args:
        persona: Reviewer persona
        journal_guidelines: Journal review guidelines
        
    Returns:
        Review instructions text
    """
    # Start with persona characteristics
    instructions = f"""
    You are acting as a peer reviewer with the following characteristics:
    
    REVIEWER BACKGROUND:
    - Field: {persona['field']}
    - Expertise: {', '.join([exp for exp in persona['expertise'][:3] if isinstance(exp, str)])}
    - Background: {persona['background']}
    """
    
    # Add the FinePersona description if available
    if persona.get("is_fine_persona", False) and persona.get("fine_persona_description"):
        instructions += f"""
    DETAILED PERSONA DESCRIPTION:
    {persona['fine_persona_description']}
    """
    
    instructions += f"""
    REVIEW STYLE:
    - Archetype: {persona['archetype'].replace('fine_persona_', 'Specialized ') if persona['archetype'].startswith('fine_persona_') else persona['archetype']} reviewer
    - Tone: {persona['tone']}
    - Depth: {persona['depth']}
    - Focus areas: {', '.join(persona['focus_areas'][:5])}
    - Strictness level: {persona['strictness']:.1f}/1.0
    - Bias: {persona['bias']}
    - Review style: {persona['review_style']}
    """
    
    # Add common criticisms if available
    if persona.get("common_criticisms"):
        instructions += "\nYou frequently raise concerns about:\n"
        for criticism in persona["common_criticisms"]:
            instructions += f"- {criticism}\n"
    
    # Add journal guidelines section
    instructions += f"""
    JOURNAL GUIDELINES:
    {journal_guidelines}
    
    REVIEW TASK:
    Create a detailed, realistic peer review of the revised academic paper following both your reviewer persona and the journal guidelines. Your review should be structured as follows:
    
    1. Summary (1-2 paragraphs summarizing the paper)
    2. Major Comments (3-5 substantive points aligned with your persona focus areas)
    3. Minor Comments (2-4 less critical issues)
    4. Overall Assessment (recommendation aligned with your strictness level)
    
    Make your review sound authentic and realistic. Focus particularly on {', '.join(persona['focus_areas'][:3])}, as these are your main areas of concern as this type of reviewer.
    """
    
    # Add specific instructions for FinePersona reviewers
    if persona.get("is_fine_persona", False):
        instructions += """
    IMPORTANT: As a reviewer based on a rich FinePersona profile, make sure your review reflects the specific expertise, background, and academic perspective described in your persona. Your voice should be distinct and reflect your unique perspective and expertise.
    """
    
    instructions += """
    Format the review as a JSON object with these fields:
    - summary: string
    - major_comments: array of strings
    - minor_comments: array of strings
    - overall_assessment: string (one of: "Accept", "Minor Revision", "Major Revision", "Reject")
    - rating: number (1-5, where 5 is excellent)
    - reviewer_perspective: string (brief description of your perspective as this reviewer)
    """
    
    return instructions

def generate_editor_summary_instructions(personas: List[Dict[str, Any]], reviews: List[Dict[str, Any]], journal_guidelines: str) -> str:
    """Generate instructions for creating an editor summary of reviews.
    
    Args:
        personas: List of reviewer personas
        reviews: List of review outputs
        journal_guidelines: Journal review guidelines
        
    Returns:
        Editor summary instructions text
    """
    # Extract ratings and assessments
    ratings = [review.get("rating", 0) for review in reviews]
    assessments = [review.get("overall_assessment", "") for review in reviews]
    
    # Calculate average rating
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    
    # Determine consensus assessment
    assessment_counts = {}
    for assessment in assessments:
        assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
    
    consensus = max(assessment_counts.items(), key=lambda x: x[1])[0] if assessment_counts else "No consensus"
    
    # Check if any FinePersona-based reviewers are included
    has_fine_personas = any(persona.get("is_fine_persona", False) for persona in personas)
    
    # Generate instructions
    instructions = f"""
    You are acting as the editor of an academic journal. You need to write a summary of peer reviews and provide an editorial decision.
    
    REVIEWER INFORMATION:
    """
    
    for i, (persona, review) in enumerate(zip(personas, reviews), 1):
        # Format the reviewer type based on whether it's a FinePersona
        if persona.get("is_fine_persona", False):
            reviewer_type = f"Specialized academic reviewer (FinePersona-based)"
        else:
            reviewer_type = f"{persona['archetype']} reviewer"
            
        instructions += f"""
    Reviewer {i}:
    - Type: {reviewer_type}
    - Focus: {', '.join(persona['focus_areas'][:3])}
    - Assessment: {review.get('overall_assessment', 'Not specified')}
    - Rating: {review.get('rating', 0)}/5
    """
        
        # Add extra information for FinePersona-based reviewers
        if persona.get("is_fine_persona", False) and persona.get("fine_persona_description"):
            # Extract a brief description (first sentence or two)
            brief_desc = ". ".join(persona["fine_persona_description"].split(". ")[:1]) + "."
            instructions += f"- Background: {brief_desc}\n"
    
    instructions += f"""
    REVIEW STATISTICS:
    - Average rating: {avg_rating:.1f}/5
    - Consensus assessment: {consensus}
    
    JOURNAL GUIDELINES:
    {journal_guidelines}
    """
    
    # Add special note if using FinePersonas
    if has_fine_personas:
        instructions += """
    NOTE: Some reviewers are based on the FinePersonas dataset, providing specialized academic perspectives beyond traditional reviewer archetypes. This enables more nuanced and domain-specific feedback that should be carefully synthesized in your editorial decision.
    """
    
    instructions += """
    EDITORIAL TASK:
    Write a comprehensive editor's summary and decision letter based on the peer reviews. Your response should include:
    
    1. Summary of Reviews (summarize key points from all reviewers)
    2. Common Concerns (identify issues raised by multiple reviewers)
    3. Unique Insights (highlight valuable unique perspectives)
    4. Editorial Decision (based on consensus and journal standards)
    5. Revision Instructions (if applicable)
    
    Format the response as a JSON object with these fields:
    - summary_of_reviews: string
    - common_concerns: array of strings
    - unique_insights: array of strings
    - decision: string (one of: "Accept", "Minor Revision", "Major Revision", "Reject")
    - revision_instructions: array of strings (empty if decision is "Accept" or "Reject")
    - editor_comments: string (additional context for the decision)
    """
    
    return instructions

def create_editor_personas(journal_guidelines: str, field: str, personas_count: int = 4) -> List[Dict[str, Any]]:
    """Create a set of editor personas for the journal.
    
    Args:
        journal_guidelines: Journal review guidelines
        field: Academic field of the paper
        personas_count: Number of editor personas to create
        
    Returns:
        List of editor personas
    """
    editor_personas = []
    used_archetypes = set()
    
    # Editor archetypes (similar to reviewer archetypes but with editorial focus)
    editor_archetypes = {
        "methodological_gatekeeper": {
            "focus": ["methodological rigor", "experimental design", "validity", "reproducibility"],
            "tone": "authoritative",
            "depth": "detailed",
            "expertise": "research methodology",
            "bias": "prioritizes scientific soundness over novelty",
            "strictness": 0.8,
            "description": "Ensures methodological rigor and scientific validity of all published work"
        },
        "field_visionary": {
            "focus": ["field advancement", "innovation", "impact", "future directions"],
            "tone": "forward-looking",
            "depth": "conceptual",
            "expertise": "emerging research trends",
            "bias": "favors bold, innovative contributions over incremental advances",
            "strictness": 0.6,
            "description": "Seeks work that will significantly advance the field in new directions"
        },
        "balanced_mediator": {
            "focus": ["balanced assessment", "fairness", "holistic evaluation", "constructive feedback"],
            "tone": "balanced",
            "depth": "comprehensive",
            "expertise": "peer review process",
            "bias": "values comprehensive, fair assessments of papers",
            "strictness": 0.5,
            "description": "Prioritizes fair and balanced evaluation of all aspects of a paper"
        },
        "standards_enforcer": {
            "focus": ["publication standards", "formatting", "compliance", "completeness"],
            "tone": "precise",
            "depth": "thorough",
            "expertise": "academic publishing standards",
            "bias": "values adherence to publishing conventions and requirements",
            "strictness": 0.7,
            "description": "Ensures all papers meet the journal's publication standards and requirements"
        }
    }
    
    # Try to use FinePersonas for editor personas as well
    fine_personas_for_editors = []
    if HUGGINGFACE_AVAILABLE:
        try:
            fine_personas = load_fine_personas_dataset()
            if fine_personas:
                # Filter for personas that mention "editor", "journal", "chief", etc.
                for persona in fine_personas:
                    description = persona.get("description", "").lower()
                    if any(term in description for term in ["editor", "journal", "editorial", "publishing", "chief", "reviewer", "committee"]):
                        fine_personas_for_editors.append(persona)
        except Exception as e:
            logger.warning(f"Error loading FinePersonas for editors: {e}")
    
    # Create each editor persona
    for i in range(personas_count):
        if i == 0 and fine_personas_for_editors:
            # Use a FinePersona for the first editor if available
            fine_persona = random.choice(fine_personas_for_editors)
            editor_archetype_id = f"fine_persona_editor_{fine_persona['id']}"
            
            # Create a persona dictionary
            persona = {
                "archetype": editor_archetype_id,
                "field": field,
                "focus_areas": fine_persona["focus_areas"],
                "tone": fine_persona["tone"],
                "depth": fine_persona["depth"],
                "expertise": fine_persona.get("expertise", ["academic publishing"]),
                "strictness": fine_persona["strictness"],
                "bias": fine_persona["bias"],
                "is_fine_persona": True,
                "fine_persona_description": fine_persona["description"],
                "persona_index": i + 1,
                "is_primary": (i == 0)
            }
            
            # Extract background from the FinePersona description
            background_parts = []
            for sentence in fine_persona["description"].split(". "):
                if any(keyword in sentence.lower() for keyword in ["editor", "journal", "review", "background", "experience", "specialized", "expert"]):
                    background_parts.append(sentence)
            
            if background_parts:
                persona["background"] = ". ".join(background_parts[:3]) + "."
            else:
                # Generate a background if none found in description
                years_experience = random.randint(5, 25)
                journal_tiers = ["top-tier", "respected", "specialized", "international", "peer-reviewed"]
                journal_tier = random.choice(journal_tiers)
                
                persona["background"] = f"Editor with {years_experience} years of experience at {journal_tier} journals in {field}. "
                persona["background"] += f"Known for {random.choice(['rigorous', 'fair', 'insightful', 'balanced'])} editorial decisions."
        else:
            # Use traditional editor archetypes
            available_archetypes = [k for k in editor_archetypes.keys() if k not in used_archetypes]
            if not available_archetypes:
                available_archetypes = list(editor_archetypes.keys())
            
            archetype_name = random.choice(available_archetypes)
            used_archetypes.add(archetype_name)
            archetype = editor_archetypes[archetype_name]
            
            # Create persona
            persona = {
                "archetype": archetype_name,
                "field": field,
                "focus_areas": archetype["focus"],
                "tone": archetype["tone"],
                "depth": archetype["depth"],
                "expertise": [archetype["expertise"]],
                "strictness": archetype["strictness"],
                "bias": archetype["bias"],
                "is_fine_persona": False,
                "persona_index": i + 1,
                "is_primary": (i == 0)
            }
            
            # Generate a background
            years_experience = random.randint(5, 25)
            journal_tiers = ["top-tier", "respected", "specialized", "international", "peer-reviewed"]
            journal_tier = random.choice(journal_tiers)
            
            persona["background"] = f"Editor with {years_experience} years of experience at {journal_tier} journals in {field}. "
            
            if persona["strictness"] > 0.7:
                persona["background"] += "Known for maintaining high standards in published work. "
            elif persona["strictness"] < 0.4:
                persona["background"] += "Known for nurturing promising work through the review process. "
            
            expertise_str = ", ".join([exp for exp in persona["expertise"] if isinstance(exp, str)])
            if expertise_str:
                persona["background"] += f"Specialized in {expertise_str}."
        
        # Generate editorial style
        persona["editorial_style"] = f"Makes editorial decisions in a {persona['tone']} tone with {persona['depth']} depth. "
        
        focus_str = ", ".join(persona["focus_areas"][:3])
        persona["editorial_style"] += f"Particularly attentive to {focus_str}. "
        
        if persona["strictness"] > 0.8:
            persona["editorial_style"] += "Very selective about what receives a positive decision."
        elif persona["strictness"] > 0.6:
            persona["editorial_style"] += "Maintains high standards for acceptance."
        elif persona["strictness"] > 0.4:
            persona["editorial_style"] += "Balances rigorous standards with openness to diverse contributions."
        else:
            persona["editorial_style"] += "Focuses on identifying promising work even if imperfect."
        
        editor_personas.append(persona)
    
    return editor_personas

def create_review_report(
    revised_paper_path: str, 
    original_comments: List[Dict[str, Any]], 
    journal_guidelines: str, 
    field: str,
    model_client,
    personas_per_reviewer: int = 4,
    editor_personas_count: int = 4,
    workflow_db=None,
    run_id: str = None
) -> Dict[str, Any]:
    """Create a comprehensive review report including reviewer feedback and editor summary.
    
    Args:
        revised_paper_path: Path to the revised paper
        original_comments: Original reviewer comments
        journal_guidelines: Journal review guidelines
        field: Academic field of the paper
        model_client: LLM client for generating reviews
        personas_per_reviewer: Number of personas per reviewer
        editor_personas_count: Number of editor personas
        workflow_db: Optional WorkflowDB instance for storing in database
        run_id: Optional run ID for database storage
        
    Returns:
        Dictionary with review report data
    """
    # Generate reviewer personas - now grouped by reviewer
    reviewer_personas_groups = create_reviewer_personas(
        original_comments, 
        journal_guidelines, 
        field,
        personas_per_reviewer=personas_per_reviewer
    )
    
    # Generate editor personas
    editor_personas = create_editor_personas(
        journal_guidelines, 
        field,
        personas_count=editor_personas_count
    )
    
    # Store personas in database if provided
    persona_id_map = {}  # Map to track persona IDs for later use
    if workflow_db and run_id:
        logger.info(f"Storing personas in database for run {run_id}")
        
        # Store reviewer personas
        for reviewer_group in reviewer_personas_groups:
            reviewer_id = reviewer_group["reviewer_id"]
            
            for persona in reviewer_group["personas"]:
                # Store in database
                persona_id = workflow_db.store_reviewer_persona(run_id, reviewer_id, persona)
                
                # Map for later use
                key = f"reviewer_{reviewer_id}_persona_{persona['persona_index']}"
                persona_id_map[key] = persona_id
        
        # Store editor personas
        for persona in editor_personas:
            # Store in database
            persona_id = workflow_db.store_editor_persona(run_id, persona)
            
            # Map for later use
            key = f"editor_persona_{persona['persona_index']}"
            persona_id_map[key] = persona_id
    
    # Load revised paper content
    with open(revised_paper_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    # Generate reviews for each reviewer (using all their personas)
    all_reviews = []
    flattened_personas = []  # Keep a flattened list of personas for the report
    
    for reviewer_group in reviewer_personas_groups:
        reviewer_id = reviewer_group["reviewer_id"]
        personas = reviewer_group["personas"]
        reviewer_reviews = []
        
        # Add personas to flattened list
        flattened_personas.extend(personas)
        
        # Generate review for each persona of this reviewer
        for persona in personas:
            # Create instructions for this persona
            instructions = generate_review_instructions(persona, journal_guidelines)
            
            # Generate review using the model
            review_json = model_client.get_completion(
                prompt=f"PAPER CONTENT:\n{paper_content}\n\nREVIEW INSTRUCTIONS:\n{instructions}",
                system_prompt="You are an academic reviewer assistant that creates realistic peer reviews based on reviewer personas and journal guidelines. Output only valid JSON.",
                max_tokens=2000
            )
            
            try:
                review = json.loads(review_json)
                review["persona_index"] = persona["persona_index"]
                review["is_primary"] = persona["is_primary"]
                reviewer_reviews.append(review)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse review JSON for persona {persona['persona_index']}: {review_json[:100]}...")
                # Create a fallback review
                reviewer_reviews.append({
                    "summary": "The paper presents interesting work but has some issues that need to be addressed.",
                    "major_comments": ["The methodology needs more clarification.", 
                                     "The literature review is incomplete."],
                    "minor_comments": ["Some grammatical errors throughout the text.",
                                     "Figures could be improved for clarity."],
                    "overall_assessment": "Major Revision",
                    "rating": 3,
                    "reviewer_perspective": f"Reviewing as a {persona['archetype']} in {field}",
                    "persona_index": persona["persona_index"],
                    "is_primary": persona["is_primary"]
                })
        
        # Add this reviewer's reviews to the overall list
        all_reviews.append({
            "reviewer_id": reviewer_id,
            "reviews": reviewer_reviews
        })
        
        # Store reviews in database if provided
        if workflow_db and run_id:
            for review in reviewer_reviews:
                # Get the persona ID from our map
                key = f"reviewer_{reviewer_id}_persona_{review['persona_index']}"
                if key in persona_id_map:
                    persona_id = persona_id_map[key]
                    
                    # Store the review
                    workflow_db.store_review(run_id, reviewer_id, persona_id, review)
    
    # Generate a consolidated review for each reviewer
    consolidated_reviews = []
    
    for reviewer_data in all_reviews:
        reviewer_id = reviewer_data["reviewer_id"]
        reviews = reviewer_data["reviews"]
        
        # Extract all assessments and ratings
        assessments = [r.get("overall_assessment", "") for r in reviews]
        ratings = [r.get("rating", 0) for r in reviews]
        
        # Determine most common assessment and average rating
        assessment_counts = {}
        for assessment in assessments:
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        most_common_assessment = max(assessment_counts.items(), key=lambda x: x[1])[0] if assessment_counts else "Major Revision"
        avg_rating = sum(ratings) / len(ratings) if ratings else 3
        
        # Collect all comments
        all_major_comments = []
        all_minor_comments = []
        
        for review in reviews:
            all_major_comments.extend(review.get("major_comments", []))
            all_minor_comments.extend(review.get("minor_comments", []))
        
        # Create consolidated review with emphasis on the primary persona
        primary_review = next((r for r in reviews if r.get("is_primary", False)), reviews[0])
        
        consolidated_review = {
            "reviewer_id": reviewer_id,
            "summary": primary_review.get("summary", ""),
            "major_comments": list(set(all_major_comments))[:5],  # Limit to top 5 unique comments
            "minor_comments": list(set(all_minor_comments))[:4],  # Limit to top 4 unique comments
            "overall_assessment": most_common_assessment,
            "rating": round(avg_rating, 1),
            "reviewer_perspective": primary_review.get("reviewer_perspective", "")
        }
        
        consolidated_reviews.append(consolidated_review)
    
    # Generate editor summaries from each editor persona
    editor_summaries = []
    
    for persona in editor_personas:
        # Create instructions for this editor persona
        editor_instructions = generate_editor_summary_instructions([p for p in flattened_personas if p.get("is_primary", False)], 
                                                               consolidated_reviews, 
                                                               journal_guidelines)
        
        # Add editor persona information to the instructions
        editor_specific_instructions = f"""
        EDITOR PERSONA:
        You are an editor with the following characteristics:
        - Background: {persona['background']}
        - Editorial style: {persona['editorial_style']}
        - Focus areas: {', '.join(persona['focus_areas'][:3])}
        - Strictness level: {persona['strictness']:.1f}/1.0
        
        Your editorial decisions should reflect this persona. {persona['bias']}
        """
        
        complete_instructions = editor_instructions + "\n\n" + editor_specific_instructions
        
        # Generate editor summary using the model
        editor_summary_json = model_client.get_completion(
            prompt=complete_instructions,
            system_prompt="You are an academic journal editor providing a summary of peer reviews and an editorial decision. Output only valid JSON.",
            max_tokens=1500
        )
        
        try:
            editor_summary = json.loads(editor_summary_json)
            editor_summary["persona_index"] = persona["persona_index"]
            editor_summary["is_primary"] = persona["is_primary"]
            editor_summaries.append(editor_summary)
            
            # Store in database if provided
            if workflow_db and run_id:
                # Get the persona ID from our map
                key = f"editor_persona_{persona['persona_index']}"
                if key in persona_id_map:
                    persona_id = persona_id_map[key]
                    
                    # Store the decision
                    workflow_db.store_editor_decision(run_id, persona_id, editor_summary)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse editor summary JSON: {editor_summary_json[:100]}...")
            # Create a fallback summary
            fallback_summary = {
                "summary_of_reviews": "The reviews highlight both strengths and weaknesses in the paper.",
                "common_concerns": ["Methodology issues", "Incomplete literature review"],
                "unique_insights": ["The paper makes a novel contribution to the field"],
                "decision": "Major Revision",
                "revision_instructions": ["Address the methodology concerns", 
                                       "Expand the literature review",
                                       "Improve the clarity of figures"],
                "editor_comments": "The paper shows promise but requires significant revisions.",
                "persona_index": persona["persona_index"],
                "is_primary": persona["is_primary"]
            }
            editor_summaries.append(fallback_summary)
            
            # Store fallback in database if provided
            if workflow_db and run_id:
                # Get the persona ID from our map
                key = f"editor_persona_{persona['persona_index']}"
                if key in persona_id_map:
                    persona_id = persona_id_map[key]
                    
                    # Store the decision
                    workflow_db.store_editor_decision(run_id, persona_id, fallback_summary)
    
    # Create a final consolidated editor decision
    # Prioritize the primary editor's decision but consider all
    decisions = [summary.get("decision", "") for summary in editor_summaries]
    decision_counts = {}
    for decision in decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    # Get the most common decision (with priority to primary editor)
    primary_editor_decision = next((s.get("decision") for s in editor_summaries if s.get("is_primary", False)), None)
    
    if primary_editor_decision:
        # Give more weight to the primary editor's decision
        decision_counts[primary_editor_decision] = decision_counts.get(primary_editor_decision, 0) + 2
    
    final_decision = max(decision_counts.items(), key=lambda x: x[1])[0] if decision_counts else "Major Revision"
    
    # Collect all common concerns and unique insights
    all_common_concerns = []
    all_unique_insights = []
    all_revision_instructions = []
    
    for summary in editor_summaries:
        all_common_concerns.extend(summary.get("common_concerns", []))
        all_unique_insights.extend(summary.get("unique_insights", []))
        if final_decision not in ["Accept", "Reject"]:
            all_revision_instructions.extend(summary.get("revision_instructions", []))
    
    # Get the primary editor's summary as the base
    primary_editor_summary = next((s for s in editor_summaries if s.get("is_primary", False)), editor_summaries[0])
    
    # Create the final consolidated editor summary
    consolidated_editor_summary = {
        "summary_of_reviews": primary_editor_summary.get("summary_of_reviews", ""),
        "common_concerns": list(set(all_common_concerns))[:5],  # Limit to top 5 unique concerns
        "unique_insights": list(set(all_unique_insights))[:5],  # Limit to top 5 unique insights
        "decision": final_decision,
        "revision_instructions": list(set(all_revision_instructions))[:7] if final_decision not in ["Accept", "Reject"] else [],
        "editor_comments": primary_editor_summary.get("editor_comments", "")
    }
    
    # Compile the complete report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "paper_path": revised_paper_path,
        "field": field,
        "reviewer_personas_groups": reviewer_personas_groups,
        "editor_personas": editor_personas,
        "all_reviews": all_reviews,
        "consolidated_reviews": consolidated_reviews,
        "editor_summaries": editor_summaries,
        "editor_summary": consolidated_editor_summary  # The final consensus summary
    }
    
    # Store the consolidated decision in the database if provided
    if workflow_db and run_id:
        workflow_db.store_consolidated_decision(run_id, consolidated_editor_summary)
        
        # Generate a process summary for use in the editor letter
        process_summary = workflow_db.get_review_process_summary(run_id)
        report["process_summary"] = process_summary
    
    return report

def save_review_report(report: Dict[str, Any], output_dir: str) -> str:
    """Save the review report to files.
    
    Args:
        report: Review report data
        output_dir: Directory to save the report
        
    Returns:
        Path to the main report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save full report as JSON
    full_report_path = os.path.join(output_dir, f"review_report_{timestamp}.json")
    with open(full_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary
    summary_path = os.path.join(output_dir, f"review_summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("PEER REVIEW SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Paper: {os.path.basename(report['paper_path'])}\n")
        f.write(f"Field: {report['field']}\n")
        
        # Add FinePersona attribution if used
        has_fine_personas = False
        for reviewer_group in report.get('reviewer_personas_groups', []):
            for persona in reviewer_group.get('personas', []):
                if persona.get("is_fine_persona", False):
                    has_fine_personas = True
                    break
            if has_fine_personas:
                break
                
        has_fine_editor = any(persona.get("is_fine_persona", False) for persona in report.get('editor_personas', []))
        
        if has_fine_personas or has_fine_editor:
            f.write("Enhanced Personas: Using FinePersonas dataset from Hugging Face\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Write editor decision
        f.write("EDITOR'S DECISION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Decision: {report['editor_summary']['decision']}\n\n")
        f.write("Summary of Reviews:\n")
        f.write(report['editor_summary']['summary_of_reviews'] + "\n\n")
        
        f.write("Common Concerns:\n")
        for concern in report['editor_summary']['common_concerns']:
            f.write(f"• {concern}\n")
        f.write("\n")
        
        f.write("Unique Insights:\n")
        for insight in report['editor_summary']['unique_insights']:
            f.write(f"• {insight}\n")
        f.write("\n")
        
        if report['editor_summary']['decision'] not in ["Accept", "Reject"]:
            f.write("Revision Instructions:\n")
            for instruction in report['editor_summary']['revision_instructions']:
                f.write(f"• {instruction}\n")
            f.write("\n")
        
        f.write("Editor's Comments:\n")
        f.write(report['editor_summary']['editor_comments'] + "\n\n")
        
        # Write information about the editorial board
        f.write("=" * 80 + "\n")
        f.write("EDITORIAL BOARD INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"This decision was made by a consensus of {len(report.get('editor_personas', []))} editors with diverse perspectives:\n\n")
        
        for i, persona in enumerate(report.get('editor_personas', []), 1):
            # Format editor type
            if persona.get("is_fine_persona", False):
                editor_type = "SPECIALIZED EDITOR"
                if "archetype" in persona and persona["archetype"].startswith("fine_persona_editor_"):
                    editor_id = persona["archetype"].replace("fine_persona_editor_", "")
                    editor_type += f" (FinePersona ID: {editor_id})"
            else:
                editor_type = persona['archetype'].upper().replace('_', ' ')
            
            primary_marker = " (PRIMARY)" if persona.get("is_primary", False) else ""
            f.write(f"Editor {i}{primary_marker}: {editor_type}\n")
            f.write(f"Background: {persona['background']}\n")
            f.write(f"Editorial Style: {persona['editorial_style']}\n")
            
            # Add the editor's decision from their summary
            editor_summary = next((s for s in report.get('editor_summaries', []) if s.get("persona_index", 0) == persona.get("persona_index", 0)), None)
            if editor_summary:
                f.write(f"Decision: {editor_summary.get('decision', 'Not specified')}\n\n")
            else:
                f.write("\n")
        
        # Write individual reviews
        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL REVIEWER ASSESSMENTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write consolidated reviews for each reviewer
        for i, review in enumerate(report['consolidated_reviews'], 1):
            reviewer_id = review['reviewer_id']
            
            f.write(f"REVIEWER {reviewer_id}: CONSOLIDATED ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Assessment: {review['overall_assessment']}\n")
            f.write(f"Rating: {review['rating']}/5\n\n")
            
            f.write("Summary:\n")
            f.write(review['summary'] + "\n\n")
            
            f.write("Major Comments:\n")
            for comment in review['major_comments']:
                f.write(f"• {comment}\n")
            f.write("\n")
            
            f.write("Minor Comments:\n")
            for comment in review['minor_comments']:
                f.write(f"• {comment}\n")
            f.write("\n")
            
            f.write("Reviewer Perspective:\n")
            f.write(review['reviewer_perspective'] + "\n\n")
            
            # Get the reviewer group for this reviewer
            reviewer_group = next((rg for rg in report.get('reviewer_personas_groups', []) if rg.get('reviewer_id') == reviewer_id), None)
            
            if reviewer_group:
                # Show information about the multiple personas
                f.write(f"NOTE: This review is a consolidation of {len(reviewer_group.get('personas', []))} different reviewer perspectives:\n\n")
                
                for j, persona in enumerate(reviewer_group.get('personas', []), 1):
                    primary_marker = " (PRIMARY)" if persona.get("is_primary", False) else ""
                    
                    # Format reviewer type based on whether it uses FinePersona
                    if persona.get("is_fine_persona", False):
                        reviewer_type = "SPECIALIZED ACADEMIC REVIEWER"
                        if "archetype" in persona and persona["archetype"].startswith("fine_persona_"):
                            reviewer_id = persona["archetype"].replace("fine_persona_", "")
                            reviewer_type += f" (ID: {reviewer_id})"
                    else:
                        reviewer_type = persona['archetype'].upper().replace('_', ' ') + " REVIEWER"
                    
                    f.write(f"Persona {j}{primary_marker}: {reviewer_type}\n")
                    
                    # For FinePersona-based reviewers, add a brief description
                    if persona.get("is_fine_persona", False) and persona.get("fine_persona_description"):
                        # Get just the first sentence
                        brief_desc = persona['fine_persona_description'].split(". ")[0] + "."
                        f.write(f"Description: {brief_desc}\n")
                    
                    f.write(f"Focus Areas: {', '.join(persona['focus_areas'][:3])}\n")
                    f.write(f"Strictness: {persona['strictness']:.1f}/1.0\n")
                    f.write("\n")
            
            if i < len(report['consolidated_reviews']):
                f.write("=" * 80 + "\n\n")
        
        # Write detailed reviews for those who want to see individual persona assessments
        f.write("=" * 80 + "\n")
        f.write("DETAILED MULTI-PERSONA REVIEW DATA\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Each reviewer used multiple personas to provide a comprehensive assessment.\n")
        f.write("Below are the individual perspectives that informed the consolidated reviews:\n\n")
        
        for reviewer_data in report.get('all_reviews', []):
            reviewer_id = reviewer_data['reviewer_id']
            reviews = reviewer_data.get('reviews', [])
            
            f.write(f"REVIEWER {reviewer_id} PERSPECTIVES:\n")
            f.write("-" * 80 + "\n\n")
            
            for review in reviews:
                persona_index = review.get('persona_index', 0)
                primary_marker = " (PRIMARY)" if review.get("is_primary", False) else ""
                
                f.write(f"Persona {persona_index}{primary_marker}:\n")
                f.write(f"Assessment: {review.get('overall_assessment', '')}\n")
                f.write(f"Rating: {review.get('rating', 0)}/5\n\n")
                
                f.write("Key Comments:\n")
                if 'major_comments' in review and review['major_comments']:
                    f.write(f"• {review['major_comments'][0]}\n")
                if len(review.get('major_comments', [])) > 1:
                    f.write(f"• {review['major_comments'][1]}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n\n")
        
        # Write footer with attribution
        f.write("=" * 80 + "\n")
        if has_fine_personas or has_fine_editor:
            f.write("This review report includes personas from the FinePersonas dataset (https://huggingface.co/datasets/argilla/FinePersonas-v0.1)\n")
        f.write("Multi-Persona Review System: Each reviewer and editor is represented by multiple perspectives\n")
        f.write("End of Review Report\n")
        f.write("=" * 80 + "\n")
    
    return summary_path