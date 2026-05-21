#!/usr/bin/env python3
"""Run the MVP acceptance test suite for hermes-memory.

Usage:
    python tests/run_acceptance_suite.py
    python tests/run_acceptance_suite.py --scenario A
    python tests/run_acceptance_suite.py --scenario K  # K requires real DBs
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
# run_acceptance_suite.py is at hermes-agent/tests/run_acceptance_suite.py
# so parent.parent = hermes-agent/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCENARIOS = {
    "A": "test_scenario_A.py",
    "B": "test_scenario_B.py",
    "C": "test_scenario_C.py",
    "D": "test_scenario_D.py",
    "E": "test_scenario_E.py",
    "F": "test_scenario_F.py",
    "G": "test_scenario_G.py",
    "H": "test_scenario_H.py",
    "I": "test_scenario_I.py",
    "J": "test_scenario_J.py",
    "K": "test_scenario_K.py",
    "L": "test_scenario_L.py",
    "M": "test_scenario_M.py",
}

SCENARIO_NAMES = {
    "A": "Lossless Capture",
    "B": "Redaction",
    "C": "Keyword Search (FTS5)",
    "D": "Semantic Search (Qdrant)",
    "E": "Hybrid Search",
    "F": "Graceful Degradation",
    "G": "Dreaming",
    "H": "Contradiction Detection",
    "I": "Provider Swap",
    "J": "Narrative Thread /new",
    "K": "Migration from Holographic",
    "L": "Backup and Restore",
    "M": "Handoff & Multi-Agent",
}

SCENARIOS_DIR = PROJECT_ROOT / "tests" / "integration" / "memory" / "scenarios"


def run_scenario(scenario: str, verbose: bool = True) -> int:
    """Run a single scenario and return exit code."""
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return 1

    filename = SCENARIOS[scenario]
    test_path = SCENARIOS_DIR / filename

    print(f"\n{'='*70}")
    print(f"  Scenario {scenario}: {SCENARIO_NAMES[scenario]}")
    print(f"{'='*70}")

    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}")
        return 1

    args = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v" if verbose else "-q",
        "--tb=short",
        "--no-header",
    ]

    result = subprocess.run(args)
    return result.returncode


def run_all(verbose: bool = True) -> dict[str, int]:
    """Run all scenarios and return {scenario: exit_code}."""
    results = {}
    for scenario in SCENARIOS:
        results[scenario] = run_scenario(scenario, verbose=verbose)
    return results


def print_summary(results: dict[str, int]) -> None:
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("  ACCEPTANCE SUITE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<6} {'Name':<35} {'Result'}")
    print(f"  {'-'*6} {'-'*35} {'-'*10}")
    for scenario, code in results.items():
        name = SCENARIO_NAMES.get(scenario, "?")
        result = "PASS" if code == 0 else "FAIL"
        print(f"  {scenario:<6} {name:<35} {result}")
    print(f"  {'-'*6} {'-'*35} {'-'*10}")

    passed = sum(1 for c in results.values() if c == 0)
    total = len(results)
    print(f"\n  {passed}/{total} scenarios passed")

    if passed == total:
        print("\n  ALL ACCEPTANCE TESTS PASSING")
    else:
        print(f"\n  {total - passed} scenario(s) need attention")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hermes-memory acceptance tests")
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        help="Run a specific scenario (A-M). If omitted, runs all.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )
    args = parser.parse_args()

    if args.scenario:
        return run_scenario(args.scenario, verbose=not args.quiet)
    else:
        results = run_all(verbose=not args.quiet)
        print_summary(results)
        return 0 if all(c == 0 for c in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())