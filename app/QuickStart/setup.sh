#!/bin/bash
# Setup script for Stock Market Prediction System

echo "=========================================="
echo "Stock Market Prediction System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo ""
echo "Installing required packages..."
echo ""

# Install packages
pip install --break-system-packages -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed"
    echo "Try installing without --break-system-packages flag:"
    echo "pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Optional: Set API keys as environment variables:"
echo "  export NEWS_API_KEY='your_newsapi_key'"
echo "  export FRED_API_KEY='your_fred_api_key'"
echo ""
echo "To get API keys:"
echo "  - News API: https://newsapi.org/"
echo "  - FRED API: https://fred.stlouisfed.org/docs/api/api_key.html"
echo ""
echo "To run the system:"
echo "  python3 main.py"
echo ""
echo "=========================================="
