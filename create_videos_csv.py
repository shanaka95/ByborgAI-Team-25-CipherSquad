import os
import csv
import random
import glob
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCSVGenerator:
    def __init__(self, 
                 data_folder: str = "data", 
                 output_file: str = "videos.csv"):
        """
        Initialize the video CSV generator.
        
        Args:
            data_folder: Path to the folder containing video files
            output_file: Path to the output CSV file
        """
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_folder)
        self.output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        
        # Define categories
        self.categories = [
            "Action", "Comedy", "Drama", "Documentary", 
            "Thriller", "Horror", "Romance", "Science Fiction", 
            "Fantasy", "Animation"
        ]
        
        # Sample descriptions for random assignment
        self.sample_descriptions = [
            "A thrilling adventure with unexpected twists and turns.",
            "A heartwarming story about friendship and loyalty.",
            "An intense drama exploring complex human relationships.",
            "A fascinating documentary about natural wonders.",
            "A spine-chilling tale of suspense and mystery.",
            "A terrifying horror story that will keep you up at night.",
            "A romantic journey of two souls finding each other.",
            "A futuristic sci-fi exploration of advanced technology.",
            "A magical fantasy world filled with mythical creatures.",
            "An animated masterpiece with stunning visuals.",
            "An action-packed sequence of daring escapes and heroic feats.",
            "A comedic take on everyday situations that will make you laugh.",
            "A dramatic portrayal of historical events that shaped our world.",
            "A documentary investigation into unsolved mysteries.",
            "A thriller that keeps you on the edge of your seat.",
            "A horror experience that explores the depths of human fear.",
            "A romantic comedy about finding love in unexpected places.",
            "A science fiction story about space exploration and alien contact.",
            "A fantasy adventure in a world of magic and wonder.",
            "An animated journey through colorful and imaginative landscapes."
        ]
    
    def get_video_files(self) -> List[str]:
        """Get a list of all video files in the data folder"""
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            pattern = os.path.join(self.data_folder, f"*{ext}")
            video_files.extend(glob.glob(pattern))
        
        # Extract just the filenames
        video_filenames = [os.path.basename(file) for file in video_files]
        logger.info(f"Found {len(video_filenames)} video files")
        
        return video_filenames
    
    def generate_random_description(self, video_name: str) -> str:
        """Generate a random description for a video"""
        # Use video name as seed for reproducibility
        random.seed(hash(video_name) % 10000)
        
        # Generate a more detailed description by combining sample descriptions
        num_sentences = random.randint(1, 3)
        selected_descriptions = random.sample(self.sample_descriptions, num_sentences)
        
        # Add video name to make it unique
        description = " ".join(selected_descriptions)
        description += f" Video ID: {video_name}."
        
        return description
    
    def generate_random_data(self, video_name: str) -> Dict[str, Any]:
        """Generate random category, watch time, and likes"""
        # Use video name as seed for reproducibility
        random.seed(hash(video_name) % 10000)
        
        return {
            "category": random.choice(self.categories),
            "watch_time": random.randint(100, 10000),  # Random watch time in seconds
            "likes": random.randint(0, 1000)  # Random number of likes
        }
    
    def create_csv(self):
        """Create the videos.csv file with all required data"""
        video_files = self.get_video_files()
        
        if not video_files:
            logger.warning(f"No video files found in {self.data_folder}")
            # Create some dummy data for testing
            video_files = [
                f"scenecliptest{i:05d}.avi" for i in range(1, 100)
            ] + [
                f"sceneclipautoautotrain{i:05d}.avi" for i in range(1, 100)
            ]
            logger.info(f"Created {len(video_files)} dummy video filenames for testing")
        
        # Create CSV file
        with open(self.output_file, 'w', newline='') as csvfile:
            fieldnames = ['video_id', 'description', 'category', 'watch_time', 'likes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Process each video
            for video_name in video_files:
                # Generate random description
                description = self.generate_random_description(video_name)
                
                # Generate random data
                random_data = self.generate_random_data(video_name)
                
                # Write to CSV
                writer.writerow({
                    'video_id': video_name,
                    'description': description,
                    'category': random_data['category'],
                    'watch_time': random_data['watch_time'],
                    'likes': random_data['likes']
                })
                
                logger.info(f"Added {video_name} to CSV")
        
        logger.info(f"CSV file created successfully: {self.output_file}")

def main():
    """Main function to create the videos.csv file"""
    generator = VideoCSVGenerator()
    generator.create_csv()

if __name__ == "__main__":
    main() 