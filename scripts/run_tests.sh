#!/bin/bash

# Shell script to run tests with common options

# Set default test directory
TEST_DIR="tests"

# Parse command line arguments
while getopts "ucifav" opt; do
  case $opt in
    u) TEST_DIR="tests/unit" ;;
    i) TEST_DIR="tests/integration" ;;
    f) TEST_DIR="tests/functional" ;;
    a) TEST_DIR="tests" ;;
    v) VERBOSE="-v" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Display help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [-u] [-i] [-f] [-a] [-v]"
    echo "  -u  Run unit tests only"
    echo "  -i  Run integration tests only"
    echo "  -f  Run functional tests only"
    echo "  -a  Run all tests (default)"
    echo "  -v  Verbose output"
    exit 0
fi

# Run the tests
echo "Running tests in $TEST_DIR"
python -m pytest $TEST_DIR $VERBOSE

# Report test results
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Check the output above for details."
fi