import os
import random
import csv
import glob
import uuid
from typing import List, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class UserVideoGenerator:
    def __init__(self, data_folder: str = "../data/"):
        """Initialize the generator with the data folder path"""
        self.data_folder = data_folder
        self.video_files = self._get_video_files()
        logger.info(f"Found {len(self.video_files)} video files")
        
    def _get_video_files(self) -> List[str]:
        """Get all video files from the data folder"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        video_files = []
        
        for ext in video_extensions:
            files = glob.glob(os.path.join(self.data_folder, f"*{ext}"))
            video_files.extend([os.path.basename(f) for f in files])
            
        if not video_files:
            raise ValueError(f"No video files found in {self.data_folder}")
            
        return video_files
        
    def generate_user_video_assignments(self, num_users: int = 10000) -> List[Dict]:
        """Generate random user-video assignments with varying number of videos per user"""
        assignments = []
        logger.info(f"Generating assignments for {num_users} users...")
        
        for _ in tqdm(range(num_users), desc="Generating user assignments"):
            # Generate random user ID
            user_id = str(uuid.uuid4())
            
            # Randomly decide how many videos this user gets (1-4)
            num_videos = random.randint(1, 4)
            
            # Randomly select videos for this user
            user_videos = random.sample(self.video_files, num_videos)
            
            # Create assignment record
            assignment = {"user_id": user_id}
            for i, video in enumerate(user_videos, 1):
                assignment[f"video{i}"] = video
            
            # Fill remaining video slots with None
            for i in range(len(user_videos) + 1, 5):
                assignment[f"video{i}"] = None
                
            assignments.append(assignment)
            
        # Calculate statistics
        video_counts = [sum(1 for k, v in assignment.items() if k.startswith('video') and v is not None)
                       for assignment in assignments]
        
        logger.info("\nAssignment Statistics:")
        logger.info(f"Total users: {len(assignments)}")
        logger.info(f"Average videos per user: {sum(video_counts) / len(assignments):.2f}")
        for i in range(1, 5):
            count = sum(1 for c in video_counts if c == i)
            percentage = (count / len(assignments)) * 100
            logger.info(f"Users with {i} videos: {count} ({percentage:.1f}%)")
            
        return assignments
        
    def save_to_csv(self, assignments: List[Dict], output_file: str = "user_video_assignments.csv"):
        """Save assignments to CSV file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to CSV
            fieldnames = ["user_id"] + [f"video{i}" for i in range(1, 5)]
            
            logger.info(f"Saving assignments to {output_file}")
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(assignments)
                
            logger.info("Assignments saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving assignments: {str(e)}")
            raise

def main():
    """Generate and save user-video assignments"""
    try:
        # Initialize generator
        generator = UserVideoGenerator()
        
        # Generate assignments
        assignments = generator.generate_user_video_assignments()
        
        # Save to CSV
        output_file = os.path.join("../data", "user_sessions.csv")
        generator.save_to_csv(assignments, output_file)
        
        # Print sample of assignments
        print("\nSample of generated assignments:")
        for assignment in assignments[:5]:
            print(f"\nUser: {assignment['user_id']}")
            for i in range(1, 5):
                video = assignment[f"video{i}"]
                if video:
                    print(f"Video {i}: {video}")
        
        print(f"\nTotal assignments generated: {len(assignments)}")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 