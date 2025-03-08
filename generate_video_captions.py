#!/usr/bin/env python3
"""
Script to generate captions for videos and store them in a SQLite3 database.
This script:
1. Creates a SQLite3 database
2. Scans for all videos in the data directory
3. Generates 10 different captions for each video
4. Stores the captions in the database along with the filename
"""

import os
import sys
import sqlite3
import random
import logging
import argparse
import subprocess
import json
from pathlib import Path
import cv2
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_captions.log")
    ]
)
logger = logging.getLogger("caption_generator")

# Caption templates to use when generating random captions
CAPTION_TEMPLATES = [
    "A video showing {adjective} {subject} {action}",
    "This video features {adjective} scenes of {subject} {action}",
    "An {adjective} clip displaying {subject} {action}",
    "Video recording of {subject} {action} in a {adjective} setting",
    "Footage of {adjective} {subject} {action} during {time_of_day}",
    "A {duration} video of {subject} {action} with {adjective} quality",
    "Clip showing {subject} {action} with {adjective} characteristics",
    "Video demonstrating {adjective} examples of {subject} {action}",
    "Recording highlighting {subject} {action} in {adjective} conditions",
    "Documentary-style footage of {subject} {action} with {adjective} features",
    "Film presenting {adjective} instances of {subject} {action}",
    "Visual representation of {subject} {action} with {adjective} elements",
]

# Words to use in caption generation
ADJECTIVES = [
    "stunning", "vibrant", "dramatic", "peaceful", "exciting", "colorful", 
    "detailed", "immersive", "high-quality", "clear", "engaging", "dynamic",
    "professional", "striking", "atmospheric", "impressive", "realistic", 
    "remarkable", "natural", "cinematic", "beautiful", "smooth", "unique"
]

SUBJECTS = [
    "landscapes", "people", "animals", "buildings", "vehicles", "events", 
    "activities", "scenes", "objects", "nature", "urban environments",
    "performances", "presentations", "artwork", "daily life", "technology"
]

ACTIONS = [
    "in motion", "interacting", "performing tasks", "on display", "in action",
    "showcasing details", "demonstrating features", "in natural settings",
    "in urban settings", "during various activities", "in different situations",
    "with various effects", "with interesting perspectives", "with unique framing"
]

TIME_OF_DAY = [
    "daytime", "nighttime", "sunset", "sunrise", "afternoon", "morning", 
    "evening", "dusk", "dawn", "midday", "twilight", "golden hour"
]

DURATIONS = [
    "short", "brief", "extended", "lengthy", "medium-length", "concise",
    "comprehensive", "detailed", "quick", "in-depth", "thorough"
]

def create_database(db_path):
    """Create a new SQLite database with the necessary tables"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create videos table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            file_path TEXT,
            duration REAL,
            width INTEGER,
            height INTEGER,
            extracted_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create captions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY,
            video_id INTEGER,
            caption_text TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
        ''')
        
        conn.commit()
        logger.info(f"Database created at {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise

def extract_video_info(video_path):
    """Extract basic information from a video file"""
    try:
        # Use OpenCV to extract video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        # Extract basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to extract text description from video (simple sample frame extraction)
        description = extract_description_from_video(cap)
        
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "description": description
        }
    except Exception as e:
        logger.error(f"Error extracting info from video {video_path}: {e}")
        return None

def extract_description_from_video(cap):
    """Extract a simple description based on a sample frame from the video"""
    try:
        # Try to capture a frame from the middle of the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count > 0:
            # Jump to the middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            
            if ret:
                # Very basic analysis - just look at brightness and color distribution
                # In a real application, you would use a more sophisticated model
                brightness = frame.mean()
                
                # Check if there's motion by comparing frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2 - 10))
                ret1, frame1 = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, frame_count // 2 + 10))
                ret2, frame2 = cap.read()
                
                motion_description = ""
                if ret1 and ret2:
                    diff = cv2.absdiff(frame1, frame2)
                    motion = diff.mean()
                    
                    if motion > 30:
                        motion_description = "high motion"
                    elif motion > 15:
                        motion_description = "moderate motion"
                    else:
                        motion_description = "low motion"
                
                # Determine color tone
                color_tone = ""
                b, g, r = cv2.split(frame)
                if r.mean() > max(b.mean(), g.mean()) + 20:
                    color_tone = "reddish"
                elif g.mean() > max(r.mean(), b.mean()) + 20:
                    color_tone = "greenish"
                elif b.mean() > max(r.mean(), g.mean()) + 20:
                    color_tone = "bluish"
                else:
                    color_tone = "balanced color"
                
                brightness_desc = ""
                if brightness < 50:
                    brightness_desc = "dark"
                elif brightness < 120:
                    brightness_desc = "moderately lit"
                else:
                    brightness_desc = "bright"
                
                return f"Video appears to be {brightness_desc} with {color_tone} tone and {motion_description}."
        
        return "No description available"
    except Exception as e:
        logger.error(f"Error extracting description: {e}")
        return "No description available"

def generate_captions(video_info, num_captions=10):
    """Generate a specified number of different captions for a video"""
    captions = []
    base_description = video_info.get("description", "")
    
    # Generate specified number of captions
    for i in range(num_captions):
        # Select random template and fill with random words
        template = random.choice(CAPTION_TEMPLATES)
        caption = template.format(
            adjective=random.choice(ADJECTIVES),
            subject=random.choice(SUBJECTS),
            action=random.choice(ACTIONS),
            time_of_day=random.choice(TIME_OF_DAY),
            duration=random.choice(DURATIONS)
        )
        
        # Add extracted info if available
        if base_description and base_description != "No description available":
            caption += f" {base_description}"
            
        # Add unique elements to make each caption different
        caption += f" Caption {i+1} of {num_captions}."
        
        captions.append(caption)
    
    return captions

def insert_video(conn, filename, file_path, video_info):
    """Insert a video record into the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO videos 
        (filename, file_path, duration, width, height, extracted_description)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            file_path,
            video_info.get("duration", 0),
            video_info.get("width", 0),
            video_info.get("height", 0),
            video_info.get("description", "")
        ))
        conn.commit()
        
        # Get the ID of the inserted record
        cursor.execute('SELECT id FROM videos WHERE filename = ?', (filename,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        logger.error(f"Error inserting video {filename}: {e}")
        conn.rollback()
        return None

def insert_captions(conn, video_id, captions):
    """Insert caption records for a video"""
    try:
        cursor = conn.cursor()
        for caption in captions:
            cursor.execute('''
            INSERT INTO captions (video_id, caption_text)
            VALUES (?, ?)
            ''', (video_id, caption))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error inserting captions for video ID {video_id}: {e}")
        conn.rollback()
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate captions for videos and store in SQLite database')
    parser.add_argument('--data-dir', default='data', help='Directory containing video files')
    parser.add_argument('--db-path', default='video_captions.db', help='Path to SQLite database')
    parser.add_argument('--captions-per-video', type=int, default=10, help='Number of captions to generate per video')
    args = parser.parse_args()
    
    # Find path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Create database
    db_path = os.path.join(script_dir, args.db_path)
    conn = create_database(db_path)
    
    try:
        # Find all video files
        video_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.webm')):
                    video_files.append((file, os.path.join(root, file)))
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Process each video
        for filename, file_path in video_files:
            logger.info(f"Processing {filename}")
            
            # Extract video information
            video_info = extract_video_info(file_path)
            if not video_info:
                logger.warning(f"Skipping {filename} - could not extract information")
                continue
            
            # Insert video record
            video_id = insert_video(conn, filename, file_path, video_info)
            if not video_id:
                logger.warning(f"Skipping {filename} - could not insert into database")
                continue
            
            # Generate captions
            captions = generate_captions(video_info, args.captions_per_video)
            
            # Insert captions
            if insert_captions(conn, video_id, captions):
                logger.info(f"Successfully processed {filename} with {len(captions)} captions")
            else:
                logger.warning(f"Failed to insert captions for {filename}")
    
    finally:
        # Close database connection
        if conn:
            conn.close()
    
    logger.info("Video caption generation complete")

if __name__ == "__main__":
    main() 