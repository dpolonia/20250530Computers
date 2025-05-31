#!/usr/bin/env python3
"""
File Preprocessing Tool for Paper Revision

This tool analyzes and preprocesses files in the 'asis' directory, converting them
to optimal formats for LLM processing. It provides model recommendations based on 
file complexity and estimates processing costs and time.

Usage:
  python preprocess_files.py [--force]

Options:
  --force    Force reprocessing of all files, even if they've been processed before
"""

import os
import sys
import time
import json
import argparse
import datetime
import mimetypes
import shutil
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import colorama
from colorama import Fore, Style
from tqdm import tqdm
from dotenv import load_dotenv

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
try:
    from src.utils.pdf_processor import PDFProcessor
    from src.utils.document_processor import DocumentProcessor
    from src.models.openai_models import get_openai_model_choices, get_openai_model_info
    from src.models.anthropic_models import get_claude_model_choices, get_claude_model_info
    from src.models.google_models import get_gemini_model_choices, get_gemini_model_info
except ImportError as e:
    print(f"{Fore.RED}Error importing required modules: {e}{Style.RESET_ALL}")
    print("Make sure you have all dependencies installed.")
    sys.exit(1)

# Constants
CACHE_DIR = "./.cache/preprocessed"
METADATA_FILE = os.path.join(CACHE_DIR, "metadata.json")
PDF_TEXT_DIR = os.path.join(CACHE_DIR, "pdf_text")
DOCX_DIR = os.path.join(CACHE_DIR, "docx")
IMAGE_DIR = os.path.join(CACHE_DIR, "images")
THUMBNAIL_DIR = os.path.join(CACHE_DIR, "thumbnails")

# Define model tiers based on complexity
MODEL_TIERS = {
    "basic": {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
        "description": "Fastest and cheapest models, good for simple documents",
        "time_multiplier": 1.0,
        "cost_multiplier": 1.0
    },
    "standard": {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4o",
        "google": "gemini-1.5-pro",
        "description": "Balanced models for most documents",
        "time_multiplier": 1.5,
        "cost_multiplier": 2.0
    },
    "advanced": {
        "anthropic": "claude-opus-4-20250514",
        "openai": "gpt-4.5-preview",
        "google": "gemini-2.5-pro-preview",
        "description": "Most powerful models for complex documents",
        "time_multiplier": 2.5,
        "cost_multiplier": 5.0
    }
}

def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * len(title)}{Style.RESET_ALL}")

def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PDF_TEXT_DIR, exist_ok=True)
    os.makedirs(DOCX_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def load_metadata() -> Dict[str, Any]:
    """Load file metadata from cache."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load metadata: {e}. Creating new metadata.{Style.RESET_ALL}")
    
    # Initialize empty metadata
    return {
        "last_processed": None,
        "files": {},
        "complexity_score": 0,
        "total_tokens": 0,
        "recommended_tier": "standard",
        "model_recommendations": {}
    }

def save_metadata(metadata: Dict[str, Any]) -> None:
    """Save file metadata to cache."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_file_mime_type(file_path: str) -> str:
    """Get the MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def process_pdf(file_path: str, file_key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process a PDF file."""
    file_info = {
        "original_path": file_path,
        "type": "pdf",
        "last_modified": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "processed": False,
        "text_path": None,
        "docx_path": None,
        "page_count": 0,
        "estimated_tokens": 0
    }
    
    try:
        # Extract text and other information from PDF
        pdf_processor = PDFProcessor(file_path)
        page_count = pdf_processor.get_page_count()
        file_info["page_count"] = page_count
        
        # Extract text
        text_output_path = os.path.join(PDF_TEXT_DIR, f"{file_key}.txt")
        pdf_processor.extract_text(text_output_path)
        file_info["text_path"] = text_output_path
        
        # Convert to DOCX for easier processing
        docx_output_path = os.path.join(DOCX_DIR, f"{file_key}.docx")
        pdf_processor.pdf_to_docx(docx_output_path)
        file_info["docx_path"] = docx_output_path
        
        # Estimate tokens (rough estimate: ~250 tokens per page)
        file_info["estimated_tokens"] = page_count * 250
        
        # Generate thumbnail of first page
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{file_key}.png")
        pdf_processor.save_first_page_as_image(thumbnail_path)
        file_info["thumbnail_path"] = thumbnail_path
        
        file_info["processed"] = True
        pdf_processor.close()
        
    except Exception as e:
        print(f"{Fore.RED}Error processing PDF {file_path}: {e}{Style.RESET_ALL}")
        file_info["error"] = str(e)
    
    return file_info

def process_docx(file_path: str, file_key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process a DOCX file."""
    file_info = {
        "original_path": file_path,
        "type": "docx",
        "last_modified": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "processed": False,
        "text_path": None,
        "estimated_tokens": 0
    }
    
    try:
        # Process DOCX file
        doc_processor = DocumentProcessor(file_path)
        
        # Extract text
        text_output_path = os.path.join(PDF_TEXT_DIR, f"{file_key}.txt")
        doc_processor.extract_text(text_output_path)
        file_info["text_path"] = text_output_path
        
        # Copy DOCX file
        docx_output_path = os.path.join(DOCX_DIR, f"{file_key}.docx")
        shutil.copy2(file_path, docx_output_path)
        file_info["docx_path"] = docx_output_path
        
        # Estimate token count based on text length (rough estimate: 1 token per 4 characters)
        with open(text_output_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            file_info["estimated_tokens"] = len(text) // 4
        
        file_info["processed"] = True
        
    except Exception as e:
        print(f"{Fore.RED}Error processing DOCX {file_path}: {e}{Style.RESET_ALL}")
        file_info["error"] = str(e)
    
    return file_info

def process_image(file_path: str, file_key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process an image file."""
    file_info = {
        "original_path": file_path,
        "type": "image",
        "last_modified": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "processed": False,
        "image_path": None,
        "estimated_tokens": 1000  # Base token estimate for images
    }
    
    try:
        # Copy image to image directory
        image_output_path = os.path.join(IMAGE_DIR, os.path.basename(file_path))
        shutil.copy2(file_path, image_output_path)
        file_info["image_path"] = image_output_path
        
        # Create thumbnail
        # In a real implementation, you would resize the image
        thumbnail_path = os.path.join(THUMBNAIL_DIR, os.path.basename(file_path))
        shutil.copy2(file_path, thumbnail_path)
        file_info["thumbnail_path"] = thumbnail_path
        
        file_info["processed"] = True
        
    except Exception as e:
        print(f"{Fore.RED}Error processing image {file_path}: {e}{Style.RESET_ALL}")
        file_info["error"] = str(e)
    
    return file_info

def process_text(file_path: str, file_key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process a text file."""
    file_info = {
        "original_path": file_path,
        "type": "text",
        "last_modified": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "processed": False,
        "text_path": None,
        "estimated_tokens": 0
    }
    
    try:
        # Copy text file
        text_output_path = os.path.join(PDF_TEXT_DIR, os.path.basename(file_path))
        shutil.copy2(file_path, text_output_path)
        file_info["text_path"] = text_output_path
        
        # Estimate tokens (rough estimate: 1 token per 4 characters)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            file_info["estimated_tokens"] = len(text) // 4
        
        file_info["processed"] = True
        
    except Exception as e:
        print(f"{Fore.RED}Error processing text file {file_path}: {e}{Style.RESET_ALL}")
        file_info["error"] = str(e)
    
    return file_info

def process_bib(file_path: str, file_key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process a BibTeX file."""
    file_info = {
        "original_path": file_path,
        "type": "bib",
        "last_modified": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "processed": False,
        "text_path": None,
        "reference_count": 0,
        "estimated_tokens": 0
    }
    
    try:
        # Copy bib file
        bib_output_path = os.path.join(PDF_TEXT_DIR, os.path.basename(file_path))
        shutil.copy2(file_path, bib_output_path)
        file_info["text_path"] = bib_output_path
        
        # Count references (rough count: each "@" usually starts a reference)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            reference_count = content.count('@')
            file_info["reference_count"] = reference_count
            
            # Estimate tokens (rough estimate: 100 tokens per reference + 1 token per 4 characters for the rest)
            file_info["estimated_tokens"] = (reference_count * 100) + (len(content) // 4)
        
        file_info["processed"] = True
        
    except Exception as e:
        print(f"{Fore.RED}Error processing BibTeX file {file_path}: {e}{Style.RESET_ALL}")
        file_info["error"] = str(e)
    
    return file_info

def calculate_complexity_score(metadata: Dict[str, Any]) -> float:
    """Calculate overall complexity score based on file properties."""
    total_tokens = 0
    total_pages = 0
    total_files = 0
    has_images = False
    reference_count = 0
    
    for file_key, file_info in metadata["files"].items():
        if file_info.get("processed", False):
            total_tokens += file_info.get("estimated_tokens", 0)
            total_pages += file_info.get("page_count", 0)
            total_files += 1
            
            if file_info.get("type") == "image":
                has_images = True
            
            if file_info.get("type") == "bib":
                reference_count += file_info.get("reference_count", 0)
    
    # Calculate complexity score (1-5 scale)
    # Factors:
    # - Total tokens (more tokens = more complex)
    # - Total pages (more pages = more complex)
    # - Has images (multimodal = more complex)
    # - Reference count (more references = more complex)
    
    token_score = min(5, total_tokens / 10000)  # 50k tokens = max score
    page_score = min(5, total_pages / 20)       # 100 pages = max score
    image_score = 1 if has_images else 0
    reference_score = min(5, reference_count / 20)  # 100 references = max score
    
    # Weighted average
    complexity_score = (
        (token_score * 0.4) +    # 40% weight for tokens
        (page_score * 0.3) +     # 30% weight for pages
        (image_score * 0.1) +    # 10% weight for images
        (reference_score * 0.2)  # 20% weight for references
    )
    
    # Update metadata
    metadata["complexity_score"] = round(complexity_score, 2)
    metadata["total_tokens"] = total_tokens
    metadata["total_pages"] = total_pages
    metadata["has_images"] = has_images
    metadata["reference_count"] = reference_count
    
    return complexity_score

def determine_recommended_tier(complexity_score: float) -> str:
    """Determine recommended model tier based on complexity score."""
    if complexity_score < 1.5:
        return "basic"
    elif complexity_score < 3.5:
        return "standard"
    else:
        return "advanced"

def generate_model_recommendations(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate model recommendations based on complexity score."""
    complexity_score = metadata["complexity_score"]
    total_tokens = metadata["total_tokens"]
    recommended_tier = determine_recommended_tier(complexity_score)
    
    # Update metadata with recommended tier
    metadata["recommended_tier"] = recommended_tier
    
    # Generate recommendations for each provider
    recommendations = {}
    for provider in ["anthropic", "openai", "google"]:
        provider_recommendations = []
        
        for tier_name, tier_info in MODEL_TIERS.items():
            model_name = tier_info[provider]
            
            # Get model info
            if provider == "anthropic":
                model_info = get_claude_model_info(model_name)
            elif provider == "openai":
                model_info = get_openai_model_info(model_name)
            else:  # google
                model_info = get_gemini_model_info(model_name)
            
            # Calculate estimated cost (handle None for model_info)
            if model_info is None:
                # Use default values if model info is not available
                input_cost = 0.001 * (total_tokens / 1000)
                output_cost = 0.002 * (total_tokens / 2000)  # Assume output is half of input
            else:
                input_cost = model_info.get("price_per_1k_input", 0.001) * (total_tokens / 1000)
                output_cost = model_info.get("price_per_1k_output", 0.002) * (total_tokens / 2000)  # Assume output is half of input
            total_cost = input_cost + output_cost
            
            # Calculate suitability score (1-5)
            # Higher score = more suitable
            tier_suitability = {
                "basic": 5 if complexity_score < 1.5 else (3 if complexity_score < 2.5 else 1),
                "standard": 3 if complexity_score < 1.5 else (5 if complexity_score < 3.5 else 3),
                "advanced": 1 if complexity_score < 2.5 else (3 if complexity_score < 3.5 else 5)
            }
            
            suitability_score = tier_suitability[tier_name]
            
            # Estimate processing time (in minutes)
            # This is just a rough estimate based on tokens and model speed
            base_time = total_tokens / 10000  # Base time in minutes
            time_estimate = base_time * tier_info["time_multiplier"]
            
            provider_recommendations.append({
                "model_name": model_name,
                "tier": tier_name,
                "suitability_score": suitability_score,
                "estimated_cost": round(total_cost, 2),
                "estimated_time_minutes": round(time_estimate, 1),
                "is_recommended": tier_name == recommended_tier
            })
        
        # Sort by suitability score (descending)
        provider_recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        recommendations[provider] = provider_recommendations
    
    # Update metadata with recommendations
    metadata["model_recommendations"] = recommendations
    
    return recommendations

def print_model_recommendations(metadata: Dict[str, Any]) -> None:
    """Print model recommendations in a user-friendly format."""
    print_section("Model Recommendations")
    
    complexity_score = metadata["complexity_score"]
    recommended_tier = metadata["recommended_tier"]
    total_tokens = metadata["total_tokens"]
    
    print(f"Document Complexity: {Fore.YELLOW}{complexity_score:.2f}/5.0{Style.RESET_ALL}")
    print(f"Estimated Total Tokens: {Fore.YELLOW}{total_tokens:,}{Style.RESET_ALL}")
    print(f"Recommended Model Tier: {Fore.GREEN}{recommended_tier.capitalize()}{Style.RESET_ALL}")
    print(f"  {MODEL_TIERS[recommended_tier]['description']}")
    
    print("\nModel Recommendations by Provider:")
    
    for provider, recommendations in metadata["model_recommendations"].items():
        print(f"\n{Fore.BLUE}{provider.upper()}{Style.RESET_ALL}")
        
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3 recommendations
            suitability = "●" * rec["suitability_score"] + "○" * (5 - rec["suitability_score"])
            cost = f"${rec['estimated_cost']:.2f}"
            time = f"{rec['estimated_time_minutes']:.1f} min"
            
            if rec["is_recommended"]:
                print(f"{Fore.GREEN}► {i}. {rec['model_name']}{Style.RESET_ALL}")
            else:
                print(f"  {i}. {rec['model_name']}")
                
            print(f"     Suitability: {Fore.YELLOW}{suitability}{Style.RESET_ALL} | " +
                  f"Cost: {Fore.YELLOW}{cost}{Style.RESET_ALL} | " +
                  f"Time: {Fore.YELLOW}{time}{Style.RESET_ALL}")

def process_files(force: bool = False) -> Dict[str, Any]:
    """Process all files in the asis directory."""
    print_header("FILE PREPROCESSING TOOL")
    
    # Ensure directories exist
    ensure_directories()
    
    # Load metadata
    metadata = load_metadata()
    
    # Get files in asis directory
    asis_dir = "./asis"
    if not os.path.isdir(asis_dir):
        print(f"{Fore.RED}Error: asis directory not found.{Style.RESET_ALL}")
        sys.exit(1)
    
    files = [f for f in os.listdir(asis_dir) if os.path.isfile(os.path.join(asis_dir, f)) and not f.endswith('.Identifier')]
    
    if not files:
        print(f"{Fore.YELLOW}No files found in asis directory.{Style.RESET_ALL}")
        return metadata
    
    print(f"Found {len(files)} files in asis directory.")
    
    # Process files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with tqdm(total=len(files), desc="Processing files") as pbar:
        for file_name in files:
            file_path = os.path.join(asis_dir, file_name)
            file_key = os.path.splitext(file_name)[0]  # Use filename without extension as key
            
            # Check if file already processed and not forcing reprocessing
            if (not force and file_key in metadata["files"] and 
                metadata["files"][file_key].get("processed", False) and 
                metadata["files"][file_key].get("last_modified") == os.path.getmtime(file_path)):
                skipped_count += 1
                pbar.update(1)
                continue
            
            # Process file based on type
            mime_type = get_file_mime_type(file_path)
            
            if mime_type == "application/pdf":
                file_info = process_pdf(file_path, file_key, metadata)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                file_info = process_docx(file_path, file_key, metadata)
            elif mime_type and mime_type.startswith("image/"):
                file_info = process_image(file_path, file_key, metadata)
            elif mime_type and mime_type.startswith("text/"):
                file_info = process_text(file_path, file_key, metadata)
            elif file_name.endswith(".bib"):
                file_info = process_bib(file_path, file_key, metadata)
            else:
                # Unknown file type, just copy
                file_info = {
                    "original_path": file_path,
                    "type": "unknown",
                    "last_modified": os.path.getmtime(file_path),
                    "size": os.path.getsize(file_path),
                    "processed": False,
                    "error": f"Unknown file type: {mime_type}"
                }
                error_count += 1
            
            # Update metadata
            metadata["files"][file_key] = file_info
            
            if file_info.get("processed", False):
                processed_count += 1
            else:
                error_count += 1
            
            pbar.update(1)
    
    # Calculate complexity score and generate recommendations
    calculate_complexity_score(metadata)
    generate_model_recommendations(metadata)
    
    # Update metadata
    metadata["last_processed"] = datetime.datetime.now().isoformat()
    save_metadata(metadata)
    
    # Print summary
    print(f"\n{Fore.GREEN}Preprocessing complete:{Style.RESET_ALL}")
    print(f"  - {processed_count} files processed")
    print(f"  - {skipped_count} files skipped (already processed)")
    print(f"  - {error_count} files with errors")
    
    # Print model recommendations
    print_model_recommendations(metadata)
    
    return metadata

def choose_model_interactive(metadata: Dict[str, Any]) -> Tuple[str, str]:
    """Interactive model selection based on recommendations."""
    print_section("Model Selection")
    
    # Choose provider
    print("Available providers:")
    providers = []
    
    for i, provider in enumerate(["anthropic", "openai", "google"], 1):
        if provider in metadata["model_recommendations"]:
            providers.append(provider)
            recommended_model = metadata["model_recommendations"][provider][0]["model_name"]
            print(f"{i}. {provider.capitalize()} (recommended: {recommended_model})")
    
    if not providers:
        print(f"{Fore.RED}No providers with recommendations available.{Style.RESET_ALL}")
        return None, None
    
    provider_choice = input("\nChoose provider (enter number or name): ")
    
    # Parse provider choice
    selected_provider = None
    try:
        if provider_choice.isdigit():
            idx = int(provider_choice) - 1
            if 0 <= idx < len(providers):
                selected_provider = providers[idx]
        else:
            provider_choice = provider_choice.lower()
            if provider_choice in providers:
                selected_provider = provider_choice
    except:
        pass
    
    if not selected_provider:
        print(f"{Fore.RED}Invalid provider choice. Using default (anthropic).{Style.RESET_ALL}")
        selected_provider = "anthropic" if "anthropic" in providers else providers[0]
    
    # Choose model
    print(f"\nModels for {selected_provider.capitalize()}:")
    recommendations = metadata["model_recommendations"].get(selected_provider, [])
    
    for i, rec in enumerate(recommendations, 1):
        suitability = "●" * rec["suitability_score"] + "○" * (5 - rec["suitability_score"])
        cost = f"${rec['estimated_cost']:.2f}"
        time = f"{rec['estimated_time_minutes']:.1f} min"
        
        if rec["is_recommended"]:
            print(f"{Fore.GREEN}► {i}. {rec['model_name']}{Style.RESET_ALL}")
        else:
            print(f"  {i}. {rec['model_name']}")
            
        print(f"     Suitability: {Fore.YELLOW}{suitability}{Style.RESET_ALL} | " +
              f"Cost: {Fore.YELLOW}{cost}{Style.RESET_ALL} | " +
              f"Time: {Fore.YELLOW}{time}{Style.RESET_ALL}")
    
    model_choice = input("\nChoose model (enter number or name, default is recommended): ")
    
    # Parse model choice
    selected_model = None
    try:
        if model_choice.isdigit():
            idx = int(model_choice) - 1
            if 0 <= idx < len(recommendations):
                selected_model = recommendations[idx]["model_name"]
        elif model_choice:
            # Find by partial name match
            for rec in recommendations:
                if model_choice.lower() in rec["model_name"].lower():
                    selected_model = rec["model_name"]
                    break
    except:
        pass
    
    if not selected_model and recommendations:
        # Use recommended model
        selected_model = recommendations[0]["model_name"]
    
    print(f"\n{Fore.GREEN}Selected model: {selected_provider.capitalize()} - {selected_model}{Style.RESET_ALL}")
    
    return selected_provider, selected_model

def main():
    """Main entry point for the preprocessing tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="File Preprocessing Tool for Paper Revision")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Process files
    metadata = process_files(force=args.force)
    
    # Ask if user wants to choose a model now
    choice = input(f"\n{Fore.YELLOW}Do you want to select a model now? (y/n): {Style.RESET_ALL}")
    
    if choice.lower() == 'y':
        provider, model = choose_model_interactive(metadata)
        
        if provider and model:
            # Save the choice to a file for paper_revision.py to use
            choice_file = os.path.join(CACHE_DIR, "model_choice.json")
            with open(choice_file, 'w') as f:
                json.dump({
                    "provider": provider,
                    "model": model,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"\n{Fore.GREEN}Model choice saved. Run paper_revision.py to start the revision process.{Style.RESET_ALL}")
            print(f"Saved command: --provider {provider} --model {model}")
    else:
        print(f"\n{Fore.BLUE}You can choose a model later when running paper_revision.py{Style.RESET_ALL}")
        print(f"Recommended: --provider {metadata['recommended_tier']} tier")

if __name__ == "__main__":
    main()