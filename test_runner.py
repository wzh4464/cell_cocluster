#!/usr/bin/env python3
"""
Simple test runner to test our modules with proper imports.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import pytest

if __name__ == "__main__":
    # Run tests that we know work
    test_args = [
        "tests/test_utils/",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--tb=short"
    ]
    
    exit_code = pytest.main(test_args)
    sys.exit(exit_code)