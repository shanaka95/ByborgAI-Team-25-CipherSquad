#!/bin/bash
# Script to run the batch video conversion

echo "Starting batch video conversion..."

# Navigate to the ByborgAI directory
cd "$(dirname "$0")"

# Check if Python is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Please install ffmpeg."
    echo "You can install it with: sudo apt-get install ffmpeg"
    exit 1
fi

# Create data_converted directory if it doesn't exist
mkdir -p data_converted

# Run the conversion script
echo "Running conversion script..."
$PYTHON_CMD batch_convert_videos.py

echo "Video conversion process completed."
echo "You can now run the webapp with: cd webapp && $PYTHON_CMD app.py" 