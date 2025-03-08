#!/usr/bin/env python3
"""
Batch converts all AVI videos in the data folder to MP4 format.
This script should be run from the ByborgAI directory.
"""

import os
import subprocess
import logging
import time
import concurrent.futures
from pathlib import Path
import sys
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_conversion.log")
    ]
)
logger = logging.getLogger("batch_convert")

# Configuration
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_converted")
MAX_WORKERS = 4  # Number of concurrent conversions
FFMPEG_PRESET = "fast"  # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
QUALITY_CRF = "23"  # Lower is better quality, 18-28 is reasonable range

def check_ffmpeg():
    """Check if ffmpeg is available on the system"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=2
        )
        if result.returncode != 0:
            logger.error("FFmpeg is not available or not working properly")
            return False
        
        logger.info(f"FFmpeg found: {result.stdout.splitlines()[0] if result.stdout else 'version info not available'}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("FFmpeg not found. Please install FFmpeg first.")
        return False

def convert_video(video_path, output_dir):
    """
    Convert a video file to MP4 format using ffmpeg
    
    Args:
        video_path: Path to the source video file
        output_dir: Directory to save the converted video
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create output filename
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        
        # Skip if output file already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping {video_path}, already converted")
            return (True, f"Already converted: {output_path}")
        
        # Log conversion start
        logger.info(f"Converting {video_path} to {output_path}")
        start_time = time.time()
        
        # Use ffmpeg to convert the video
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c:v', 'libx264',     # Video codec
            '-preset', FFMPEG_PRESET,  # Encoding speed/quality tradeoff
            '-crf', QUALITY_CRF,    # Quality (lower is better)
            '-c:a', 'aac',         # Audio codec
            '-b:a', '128k',        # Audio bitrate
            '-movflags', '+faststart',  # Enable streaming
            '-y',                  # Overwrite output file if it exists
            output_path
        ]
        
        # Run ffmpeg command
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error converting {video_path}: {stderr}")
            return (False, f"Conversion failed: {stderr[:100]}...")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Converted {video_path} in {elapsed_time:.2f} seconds")
        
        # Verify the converted file exists and has size > 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return (True, f"Successfully converted: {output_path}")
        else:
            return (False, "Conversion completed but output file is missing or empty")
        
    except Exception as e:
        logger.error(f"Exception converting {video_path}: {str(e)}")
        return (False, f"Exception: {str(e)}")

def main():
    # Check if ffmpeg is available
    if not check_ffmpeg():
        logger.error("FFmpeg is required but not available. Exiting.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Find all AVI files in the source directory
    video_files = list(Path(SOURCE_DIR).glob("**/*.avi"))
    
    if not video_files:
        logger.warning(f"No AVI files found in {SOURCE_DIR}")
        return
    
    logger.info(f"Found {len(video_files)} AVI files to convert")
    
    # Process files with a thread pool
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit conversion tasks
        future_to_video = {
            executor.submit(convert_video, str(video_path), TARGET_DIR): video_path
            for video_path in video_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                success, message = future.result()
                if success:
                    if "Already converted" in message:
                        results["skipped"] += 1
                    else:
                        results["success"] += 1
                else:
                    results["failed"] += 1
                    logger.error(f"Failed to convert {video_path}: {message}")
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Exception processing {video_path}: {str(e)}")
    
    # Print summary
    logger.info(f"Conversion complete. Summary:")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped (already converted): {results['skipped']}")
    logger.info(f"Converted videos are in: {TARGET_DIR}")

if __name__ == "__main__":
    main() 