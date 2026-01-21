#!/bin/bash
#
# Run all tests and quality checks locally
#
# Usage: ./run_tests.sh [options]
#
# Options:
#   --quick    Run only fast tests
#   --full     Run all tests including slow ones
#   --lint     Run only linting
#   --fix      Auto-fix linting issues

set -e

echo "🧪 scVAE-Annotator Test Runner"
echo "================================"

# Parse arguments
QUICK=false
FULL=false
LINT_ONLY=false
FIX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --full)
            FULL=true
            shift
            ;;
        --lint)
            LINT_ONLY=true
            shift
            ;;
        --fix)
            FIX=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Linting
if [ "$LINT_ONLY" = true ] || [ "$FULL" = true ]; then
    echo ""
    echo "📋 Running Code Quality Checks..."
    echo "================================"
    
    if [ "$FIX" = true ]; then
        echo "🔧 Auto-fixing with Black..."
        black .
        
        echo "🔧 Auto-fixing imports with isort..."
        isort .
    else
        echo "🎨 Checking format with Black..."
        black --check --diff . || true
        
        echo "📦 Checking imports with isort..."
        isort --check-only --diff . || true
    fi
    
    echo "🔍 Running Flake8..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics || true
    
    echo "🔬 Running Pylint..."
    pylint scvae_annotator.py src/ --exit-zero --max-line-length=127 || true
    
    echo "🛡️  Running Bandit (security)..."
    bandit -r . -f screen || true
    
    echo "⚠️  Running Safety (dependencies)..."
    safety check || true
    
    if [ "$LINT_ONLY" = true ]; then
        exit 0
    fi
fi

# Unit Tests
if [ "$QUICK" = true ]; then
    echo ""
    echo "⚡ Running Quick Tests..."
    echo "================================"
    pytest tests/ -v -m "not slow" --cov=. --cov-report=term-missing
elif [ "$FULL" = true ]; then
    echo ""
    echo "🚀 Running Full Test Suite..."
    echo "================================"
    pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml
else
    echo ""
    echo "🧪 Running Standard Tests..."
    echo "================================"
    pytest tests/ -v --cov=. --cov-report=term-missing -m "not slow"
fi

echo ""
echo "✅ All checks completed!"

if [ "$FULL" = true ]; then
    echo ""
    echo "📊 Coverage report generated in htmlcov/index.html"
fi
