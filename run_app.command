#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Starting Insider Threat Analytics..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    read -p "Press Enter to exit..."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Checking dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Launching application..."
streamlit run app.py

# Keep terminal open if app crashes
read -p "Press Enter to exit..."
