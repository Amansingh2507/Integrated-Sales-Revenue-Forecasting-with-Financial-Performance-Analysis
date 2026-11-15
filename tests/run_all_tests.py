"""
Master test runner executing all unit and integration test suites.
Can be executed directly via `python tests/run_all_tests.py` or `pytest tests/`.
"""

import unittest
import sys
from pathlib import Path

def run_all_tests():
    test_dir = Path(__file__).resolve().parent
    print("=" * 70)
    print("EXECUTING MASTER AUTOMATED TEST SUITE (DATA, ML, SQL & FP&A)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"TESTS RUN: {result.testsRun} | FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("[SUCCESS] All system tests passed with zero failures!")
        return 0
    else:
        print("[FAILURE] One or more tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
