#!/bin/bash

# Ensure script is run as: `bash setup.sh`

# Desired Python version
PYTHON_VERSION="3.9"

# ------------------------------------------------------------------------------
# Step 1: Check Python version
# ------------------------------------------------------------------------------
echo "Checking Python version..."
if ! python3 --version | grep -q "$PYTHON_VERSION"; then
  echo "ERROR: Python $PYTHON_VERSION is required. Current version is:"
  python3 --version
  echo "Please install Python $PYTHON_VERSION before running this script."
  exit 1
fi

# ------------------------------------------------------------------------------
# Step 2: Create Virtual Environment
# ------------------------------------------------------------------------------
echo "Creating virtual environment with Python $PYTHON_VERSION..."
python3 -m venv env
source env/bin/activate

# ------------------------------------------------------------------------------
# Step 3: Upgrade Pip and Install Dependencies
# ------------------------------------------------------------------------------
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Activate your environment with:"
echo "source env/bin/activate"
