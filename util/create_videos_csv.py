import os
import csv
import random
import glob
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCSVGenerator:
    def __init__(self, 
                 data_folder: str = "data", 
                 output_file: str = "videos.csv",
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333):
        """
        Initialize the video CSV generator.
        
        Args:
            data_folder: Path to the folder containing video files
            output_file: Path to the output CSV file
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port number
        """
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_folder)
        self.output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        
        # Connect to Qdrant
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.descriptions_collection = "video_descriptions"
        
        # Define categories
        self.categories = [
            "Action", "Comedy", "Drama", "Documentary", 
            "Thriller", "Horror", "Romance", "Science Fiction", 
            "Fantasy", "Animation"
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
    
    def get_video_description(self, video_name: str) -> str:
        """Retrieve the description for a video from Qdrant"""
        try:
            # Find the video's description
            video_results = self.client.scroll(
                collection_name=self.descriptions_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_name)
                        )
                    ]
                ),
                limit=1
            )
            
            points = video_results[0]
            if not points:
                logger.warning(f"No description found for video: {video_name}")
                return f"No description available for {video_name}"
            
            description = points[0].payload.get("description", f"No description available for {video_name}")
            return description
            
        except Exception as e:
            logger.error(f"Error retrieving description for {video_name}: {str(e)}")
            return f"Error retrieving description for {video_name}: {str(e)}"
    
    def generate_random_data(self) -> Dict[str, Any]:
        """Generate random category, watch time, and likes"""
        return {
            "category": random.choice(self.categories),
            "watch_time": random.randint(100, 10000),  # Random watch time in seconds
            "likes": random.randint(0, 1000)  # Random number of likes
        }
    
    def create_csv(self):
        """Create the videos.csv file with all required data"""
        video_files = self.get_video_files()
        
        # Create CSV file
        with open(self.output_file, 'w', newline='') as csvfile:
            fieldnames = ['video_id', 'description', 'category', 'watch_time', 'likes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Process each video
            for video_name in video_files:
                # Get description from vector database
                description = self.get_video_description(video_name)
                
                # Generate random data
                random_data = self.generate_random_data()
                
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