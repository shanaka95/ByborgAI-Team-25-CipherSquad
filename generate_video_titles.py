#!/usr/bin/env python3
"""
Script to generate alternative titles for videos based on their descriptions
and store them in a SQLite database using a local Llama 2 8B model.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path to import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from recommender import VideoRecommender
except ImportError:
    from mock_recommender import VideoRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_titles_generation.log")
    ]
)
logger = logging.getLogger("title_generator")

# Initialize the recommender to get video descriptions
recommender = VideoRecommender()

# Initialize model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Using chat model for better instruction following
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the Llama 2 model and tokenizer"""
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            load_in_8bit=True  # Enable 8-bit quantization to reduce memory usage
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def setup_database():
    """Create SQLite database and tables"""
    try:
        conn = sqlite3.connect('video_titles.db')
        cursor = conn.cursor()
        
        # Create table for video titles and descriptions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_titles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            original_description TEXT,
            generated_description TEXT,
            generated_title TEXT NOT NULL,
            is_generated_desc BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create index on video_id for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_video_id 
        ON video_titles(video_id)
        ''')
        
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise

def generate_description(model, tokenizer, video_id):
    """
    Generate a description for a video that doesn't have one
    """
    try:
        # Construct the prompt for description generation
        prompt = f"""Given a video with ID {video_id}, generate a general description that could be used to create engaging titles.
        The description should be informative and generic enough to be useful for title generation.
        Keep the description between 50-100 words.

        Generate a description:"""

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate description
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the description
        description = description.replace(prompt, "").strip()
        
        return description

    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        return None

def generate_titles(model, tokenizer, description, num_titles=10):
    """
    Generate alternative titles for a video based on its description
    using the local Llama 2 model
    """
    try:
        # Construct the prompt for title generation
        prompt = f"""Based on the following video description, generate {num_titles} different, 
        engaging titles that accurately represent the content. Each title should be unique, concise,
        and engaging. Format the output as a JSON array of strings.

        Description: {description}

        Generate exactly {num_titles} titles in this format:
        ["Title 1", "Title 2", "Title 3"]

        Titles:"""

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate titles
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the JSON array from the response
        match = re.search(r'\[.*\]', generated_text, re.DOTALL)
        if match:
            try:
                titles = json.loads(match.group())
                # Ensure we have exactly num_titles
                if len(titles) < num_titles:
                    # Generate more titles if needed
                    while len(titles) < num_titles:
                        titles.append(f"Engaging Video {len(titles) + 1}")
                titles = titles[:num_titles]
                return titles
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {match.group()}")
                return [f"Engaging Video {i+1}" for i in range(num_titles)]
        else:
            logger.warning(f"Could not find JSON array in response: {generated_text}")
            return [f"Engaging Video {i+1}" for i in range(num_titles)]

    except Exception as e:
        logger.error(f"Error generating titles: {str(e)}")
        return [f"Engaging Video {i+1}" for i in range(num_titles)]

def process_videos(model, tokenizer):
    """Process all videos and generate alternative titles"""
    try:
        # Set up database connection
        conn = setup_database()
        cursor = conn.cursor()
        
        # Get list of video files
        video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        video_files = list(Path(video_dir).glob("**/*.avi"))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                # Extract video ID from filename
                video_id = video_path.stem
                
                # Check if we already have titles for this video
                cursor.execute("SELECT COUNT(*) FROM video_titles WHERE video_id = ?", (video_id,))
                if cursor.fetchone()[0] >= 10:
                    logger.info(f"Skipping {video_id}, already has 10 titles")
                    continue
                
                # Get video description
                video_data = recommender.get_video_description(video_id)
                description = video_data.get("description") if "error" not in video_data else None
                
                # If no description available, generate one
                generated_description = None
                if not description:
                    logger.info(f"Generating description for {video_id}")
                    generated_description = generate_description(model, tokenizer, video_id)
                    if not generated_description:
                        logger.error(f"Could not generate description for {video_id}")
                        continue
                    description = generated_description
                
                # Generate alternative titles
                titles = generate_titles(model, tokenizer, description)
                
                # Store titles in database
                for title in titles:
                    cursor.execute("""
                    INSERT INTO video_titles (
                        video_id, 
                        original_description, 
                        generated_description,
                        generated_title,
                        is_generated_desc
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        video_id, 
                        description if not generated_description else None,
                        generated_description,
                        title,
                        1 if generated_description else 0
                    ))
                
                conn.commit()
                logger.info(f"Generated and stored {len(titles)} titles for {video_id}")
                
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
                continue
        
        # Close database connection
        conn.close()
        logger.info("Completed processing all videos")
        
    except Exception as e:
        logger.error(f"Error in process_videos: {str(e)}")
        if 'conn' in locals():
            conn.close()

def main():
    try:
        # Load model and tokenizer
        model, tokenizer = load_model()
        
        # Process videos
        process_videos(model, tokenizer)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 