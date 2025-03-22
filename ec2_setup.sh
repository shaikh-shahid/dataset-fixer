#!/bin/bash

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install python3-pip -y

# Install required Python packages
pip3 install pandas tqdm psutil

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the CodeLlama model (this might take a few minutes)
ollama pull codellama:7b

# Create working directory
mkdir -p ~/vulnerability_fixer

echo "Setup complete! Now you can:"
echo "1. Upload your CSV file to the instance"
echo "2. Upload the Python script"
echo "3. Run the script with: python3 fix_incomplete_code.py" 