# Video Caption Generator

This script automatically generates captions for video files and stores them in a SQLite database. It can process all video files in a specified directory and generate multiple captions for each video.

## Features

- Creates a SQLite database to store video information and captions
- Scans a directory for video files (AVI, MP4, MOV, MKV, WEBM)
- Extracts basic information from each video (width, height, duration)
- Analyzes video frames to create a basic description
- Generates multiple unique captions for each video
- Stores the captions in the database with the video file name

## Requirements

- Python 3.6+
- OpenCV (for video analysis)
- SQLite3 (included with Python)

## Installation

1. Install the required dependencies:

```bash
pip install opencv-python
```

2. Clone or download this repository

## Usage

Run the script with the following command:

```bash
python generate_video_captions.py --data-dir /path/to/videos --db-path /path/to/output.db --captions-per-video 10
```

### Arguments

- `--data-dir`: Directory containing the video files (default: "data")
- `--db-path`: Path where the SQLite database will be created (default: "video_captions.db")
- `--captions-per-video`: Number of captions to generate for each video (default: 10)

## Database Schema

The script creates a SQLite database with the following tables:

### Videos Table

- `id`: Integer primary key
- `filename`: Text (unique) - the filename of the video
- `file_path`: Text - the full path to the video file
- `duration`: Real - duration of the video in seconds
- `width`: Integer - width of the video in pixels
- `height`: Integer - height of the video in pixels
- `extracted_description`: Text - a description extracted from the video frames
- `created_at`: Timestamp - when the record was created

### Captions Table

- `id`: Integer primary key
- `video_id`: Integer - foreign key to the videos table
- `caption_text`: Text - the generated caption
- `generated_at`: Timestamp - when the caption was generated

## How It Works

1. The script scans the specified directory for video files
2. For each video, it extracts information using OpenCV:
   - Basic video properties (duration, dimensions, etc.)
   - A simple description based on video frame analysis
3. It generates multiple captions using templates and random word choices
4. All information is stored in the SQLite database

## Example Output

The generated captions will look something like:

- "A video showing vibrant landscapes in motion. Video appears to be bright with balanced color tone and moderate motion. Caption 1 of 10."
- "This video features stunning scenes of animals in natural settings. Video appears to be moderately lit with bluish tone and high motion. Caption 2 of 10."

## Extending the Script

You can extend this script by:

- Adding more templates and words to the caption generation
- Implementing a more sophisticated video analysis using computer vision techniques
- Integrating with a machine learning model for better descriptions
- Adding a web interface to browse and edit the generated captions 