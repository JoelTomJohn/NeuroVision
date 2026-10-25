"""
run_tests.py

Convenient test runner for NeuroVision.

Usage:
    python run_tests.py         # Runs fast unit tests (default)
    python run_tests.py --fast  # Runs fast unit tests
    python run_tests.py --all   # Runs unit tests + full SegResNet benchmark test (~90s on CPU)
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add project root to sys.path
_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))


def main():
    parser = argparse.ArgumentParser(description="NeuroVision Test Runner")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including long-running SegResNet inference benchmark",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast unit tests only (skips long inference)",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always add unit tests
    from tests import test_units
    suite.addTests(loader.loadTestsFromModule(test_units))

    if args.all:
        print("[NeuroVision] Adding full SegResNet benchmark test suite...")
        from tests import test_benchmark
        suite.addTests(loader.loadTestsFromModule(test_benchmark))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
