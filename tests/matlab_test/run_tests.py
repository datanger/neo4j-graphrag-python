#!/usr/bin/env python3
"""
Test Runner for MATLAB Code Extractor.

This script provides a convenient way to run all tests for the MATLAB code extractor,
including unit tests and integration tests. It supports different test modes and
provides detailed output and reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            print("✓ SUCCESS")
            if result.stdout:
                print(result.stdout)
        else:
            print("✗ FAILED")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/matlab_test/test_matlab_extractor.py", "--asyncio-mode=auto"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=neo4j_graphrag.experimental.components.code_extractor.matlab", "--cov-report=term-missing"])
    
    return run_command(cmd, "Unit Tests")

def run_integration_tests(verbose=False, neo4j_available=False):
    """Run integration tests."""
    if neo4j_available:
        cmd = ["python", "-m", "pytest", "tests/matlab_test/test_full_pipeline.py", "--asyncio-mode=auto"]
        if verbose:
            cmd.append("-v")
        return run_command(cmd, "Integration Tests (with Neo4j)")
    else:
        # Run only tests that don't require Neo4j
        cmd = [
            "python", "-m", "pytest", 
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_integration",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_cross_file_relationship_processing",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_data_validation_and_conversion",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_error_handling_and_recovery",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_performance_and_scalability",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_real_world_scenario",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_data_integrity_and_consistency",
            "tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_edge_cases_and_boundary_conditions",
            "--asyncio-mode=auto"
        ]
        if verbose:
            cmd.append("-v")
        return run_command(cmd, "Integration Tests (without Neo4j)")

def run_all_tests(verbose=False, coverage=False, neo4j_available=False):
    """Run all tests."""
    print("MATLAB Code Extractor Test Suite")
    print("=" * 60)
    
    # Check Neo4j availability
    if neo4j_available:
        print("✓ Neo4j database available - running full integration tests")
    else:
        print("⚠️  Neo4j database not available - running limited integration tests")
    
    # Run unit tests
    unit_success = run_unit_tests(verbose, coverage)
    
    # Run integration tests
    integration_success = run_integration_tests(verbose, neo4j_available)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Unit Tests: {'✓ PASSED' if unit_success else '✗ FAILED'}")
    print(f"Integration Tests: {'✓ PASSED' if integration_success else '✗ FAILED'}")
    
    if unit_success and integration_success:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False

def check_neo4j_availability():
    """Check if Neo4j is available."""
    try:
        import neo4j
        driver = neo4j.GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        with driver.session(database="neo4j") as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run MATLAB Code Extractor Tests")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--neo4j", action="store_true", help="Force Neo4j integration tests")
    parser.add_argument("--no-neo4j", action="store_true", help="Skip Neo4j integration tests")
    
    args = parser.parse_args()
    
    # Determine Neo4j availability
    if args.neo4j:
        neo4j_available = True
    elif args.no_neo4j:
        neo4j_available = False
    else:
        neo4j_available = check_neo4j_availability()
    
    # Run tests based on arguments
    if args.unit_only:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration_only:
        success = run_integration_tests(args.verbose, neo4j_available)
    else:
        success = run_all_tests(args.verbose, args.coverage, neo4j_available)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 