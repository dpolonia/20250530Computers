#!/usr/bin/env python3
"""
Paper Revision Tool - Main Entry Point.

This is a wrapper script that imports from the refactored modules.
"""

import sys
from src.core.paper_revision import main

if __name__ == "__main__":
    sys.exit(main())