#!/bin/bash

# Check if the input directory parameter is provided
if [ -z "$1" ]; then
    echo "Usage: graphrag_workflow.sh <inputDir>"
    exit 1
fi

# Set the inputDir variable from the first argument
inputDir="$1"

# Check if inputDir starts with "proj_"
if [[ "$(basename "$inputDir")" != proj_* ]]; then
    echo "Error: The input directory name must start with 'proj_'"
    exit 1
fi

# Create the sub-directory inputDir/input
mkdir -p "$inputDir/input"
if [ $? -ne 0 ]; then
    echo "Failed to create directory: $inputDir/input"
    exit 1
fi
echo "Created directory: $inputDir/input"

# Copy all .txt files from the input directory to inputDir/input
cp input/*.txt "$inputDir/input"
if [ $? -ne 0 ]; then
    echo "Failed to copy .txt files to $inputDir/input"
    exit 1
fi
echo "Copied all .txt files to $inputDir/input"

# Initialize the graphrag index with the specified root
python -m graphrag.index --init --root "$inputDir"
if [ $? -ne 0 ]; then
    echo "Failed to initialize graphrag index."
    exit 1
fi
echo "Initialized graphrag index in $inputDir."

# Copy settings.yaml to inputDir
cp "settings.yaml" "$inputDir"
if [ $? -ne 0 ]; then
    echo "Failed to copy settings.yaml to $inputDir."
    exit 1
fi
echo "Copied settings.yaml to $inputDir."

# Rename community_report.txt to community_reports.txt in inputDir/prompts
if [ -f "$inputDir/prompts/community_report.txt" ]; then
    mv "$inputDir/prompts/community_report.txt" "$inputDir/prompts/community_reports.txt"
    if [ $? -ne 0 ]; then
        echo "Failed to rename community_report.txt to community_reports.txt."
        exit 1
    fi
    echo "Renamed community_report.txt to community_reports.txt."
else
    echo "File community_report.txt does not exist in $inputDir/prompts."
fi

# Copy all prompt files from the prompt directory to inputDir/prompts
cp prompts/*.txt "$inputDir/prompts"
if [ $? -ne 0 ]; then
    echo "Failed to copy prompt files to $inputDir/prompts"
    exit 1
fi
echo "Copied all prompt files to $inputDir/prompt"

# Run graphrag index with the updated root
python -m graphrag.index --root "$inputDir"
if [ $? -ne 0 ]; then
    echo "Failed to run graphrag index with root $inputDir."
    exit 1
fi
echo "Successfully ran graphrag index with root $inputDir."
