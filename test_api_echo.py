#!/usr/bin/env python3
"""
API Echo Test for LLM Providers.

This script tests all three LLM providers (Anthropic, OpenAI, Google) 
by asking each for the current date, time, and location, and echoes
the response.
"""

import os
import sys
import time
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init
import anthropic
import openai
import google.generativeai as genai

# Initialize colorama
colorama_init()

# Load environment variables
load_dotenv()

def test_anthropic():
    """Test Anthropic Claude API."""
    print(f"\n{Fore.CYAN}Testing Anthropic Claude API:{Style.RESET_ALL}")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{Fore.RED}[✗] ANTHROPIC_API_KEY not found{Style.RESET_ALL}")
        return False
        
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple prompt asking for date, time, and location
        prompt = "What is the current date and time? Where are you located?"
        
        print(f"{Fore.BLUE}[*] Sending request to Claude...{Style.RESET_ALL}")
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-opus-4-20250514",  # Using the strongest Anthropic model
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}[✓] Received response in {elapsed_time:.2f}s{Style.RESET_ALL}")
        
        # Extract and print response
        answer = response.content[0].text
        print(f"\n{Fore.YELLOW}Claude's Response:{Style.RESET_ALL}")
        print(f"{answer}\n")
        
        # Print token usage
        print(f"{Fore.BLUE}Token usage:{Style.RESET_ALL}")
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")
        
        return True
    except Exception as e:
        print(f"{Fore.RED}[✗] Error: {str(e)}{Style.RESET_ALL}")
        return False

def test_openai():
    """Test OpenAI API."""
    print(f"\n{Fore.CYAN}Testing OpenAI API:{Style.RESET_ALL}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{Fore.RED}[✗] OPENAI_API_KEY not found{Style.RESET_ALL}")
        return False
        
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Simple prompt asking for date, time, and location
        prompt = "What is the current date and time? Where are you located?"
        
        print(f"{Fore.BLUE}[*] Sending request to GPT...{Style.RESET_ALL}")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4.5-preview",  # Using the strongest OpenAI model
            max_tokens=100,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}[✓] Received response in {elapsed_time:.2f}s{Style.RESET_ALL}")
        
        # Extract and print response
        answer = response.choices[0].message.content
        print(f"\n{Fore.YELLOW}GPT's Response:{Style.RESET_ALL}")
        print(f"{answer}\n")
        
        # Print token usage
        print(f"{Fore.BLUE}Token usage:{Style.RESET_ALL}")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        
        return True
    except Exception as e:
        print(f"{Fore.RED}[✗] Error: {str(e)}{Style.RESET_ALL}")
        return False

def test_google():
    """Test Google Gemini API."""
    print(f"\n{Fore.CYAN}Testing Google Gemini API:{Style.RESET_ALL}")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(f"{Fore.RED}[✗] GOOGLE_API_KEY not found{Style.RESET_ALL}")
        return False
        
    try:
        genai.configure(api_key=api_key)
        
        # Simple prompt asking for date, time, and location
        prompt = "What is the current date and time? Where are you located?"
        
        print(f"{Fore.BLUE}[*] Sending request to Gemini...{Style.RESET_ALL}")
        start_time = time.time()
        
        # Use the full model name including the 'models/' prefix
        model = genai.GenerativeModel('models/gemini-1.5-pro')  # Using the most stable Gemini model
        response = model.generate_content(prompt)
        
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}[✓] Received response in {elapsed_time:.2f}s{Style.RESET_ALL}")
        
        # Extract and print response
        answer = response.text
        print(f"\n{Fore.YELLOW}Gemini's Response:{Style.RESET_ALL}")
        print(f"{answer}\n")
        
        # Gemini doesn't provide token counts, so we estimate
        print(f"{Fore.BLUE}Token usage (estimated):{Style.RESET_ALL}")
        input_tokens = len(prompt) // 4
        output_tokens = len(answer) // 4
        print(f"Input tokens (est): {input_tokens}")
        print(f"Output tokens (est): {output_tokens}")
        print(f"Total tokens (est): {input_tokens + output_tokens}")
        
        return True
    except Exception as e:
        print(f"{Fore.RED}[✗] Error: {str(e)}{Style.RESET_ALL}")
        return False

def main():
    """Run the API echo tests."""
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}API Echo Test - Date, Time & Location (Validated Models){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    
    results = []
    
    # Test Anthropic
    anthropic_result = test_anthropic()
    results.append(("Anthropic Claude", anthropic_result))
    
    # Test OpenAI
    openai_result = test_openai()
    results.append(("OpenAI GPT", openai_result))
    
    # Test Google
    google_result = test_google()
    results.append(("Google Gemini", google_result))
    
    # Print summary
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Results Summary:{Style.RESET_ALL}")
    
    for provider, success in results:
        if success:
            print(f"{Fore.GREEN}[✓] {provider}: Successful{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[✗] {provider}: Failed{Style.RESET_ALL}")
    
    # Overall result
    if all(result[1] for result in results):
        print(f"\n{Fore.GREEN}All API tests passed successfully!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}Some API tests failed. See details above.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()