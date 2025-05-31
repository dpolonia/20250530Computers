"""
Interactive mode functionality for the paper revision tool.
"""

from colorama import Fore, Style
from typing import Optional


def interactive_wait(message: str, path: Optional[str] = None, interactive: bool = True) -> None:
    """
    Pause execution in interactive mode and wait for user input.
    
    Args:
        message: Message to display to the user
        path: Optional path to show to the user
        interactive: Whether the tool is in interactive mode
    """
    if not interactive:
        return
        
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[INTERACTIVE STOP POINT]{Style.RESET_ALL}")
    print(f"{message}")
    if path:
        print(f"\nYou can find the output at: {Fore.YELLOW}{path}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    try:
        input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
    except (EOFError, KeyboardInterrupt):
        # In case of non-interactive environment
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Non-interactive environment detected, continuing automatically")
        pass