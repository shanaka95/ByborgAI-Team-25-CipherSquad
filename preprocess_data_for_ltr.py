import os
import csv
import logging
from typing import List, Dict, Tuple, Any
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    from transformers import AutoTokenizer, AutoModel
    ADVANCED_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Some advanced dependencies are not available. Using basic functionality.")
    ADVANCED_IMPORTS_AVAILABLE = False
    
    # Define minimal CSV reading functions if pandas is not available
    def read_csv_basic(file_path):
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

class DataPreprocessor:
    """
    Preprocesses user session data for learning to rank (LTR) by calculating
    similarity metrics between video pairs.
    """
    
    def __init__(
        self,
        user_sessions_file: str = "user-sessions/user_sessions.csv",
        videos_file: str = "videos.csv",
        output_file: str = "preprocessed_user_video_data.csv"
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            user_sessions_file: Path to the user sessions CSV file
            videos_file: Path to the videos CSV file
            output_file: Path to the output preprocessed CSV file
        """
        self.user_sessions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), user_sessions_file)
        self.videos_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), videos_file)
        self.output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        
        # Initialize BERT model and tokenizer for text similarity if available
        self.bert_available = False
        if ADVANCED_IMPORTS_AVAILABLE:
            try:
                logger.info("Loading BERT model...")
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.model = AutoModel.from_pretrained('bert-base-uncased')
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                logger.info(f"BERT model loaded successfully (using {self.device})")
                self.bert_available = True
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {str(e)}")
                logger.warning("Will use random similarity scores instead")
    
    def load_data(self):
        """Load user sessions and videos data"""
        try:
            # Load user sessions and videos based on available libraries
            if ADVANCED_IMPORTS_AVAILABLE:
                user_sessions = pd.read_csv(self.user_sessions_file)
                videos = pd.read_csv(self.videos_file)
            else:
                user_sessions = read_csv_basic(self.user_sessions_file)
                videos = read_csv_basic(self.videos_file)
                
            logger.info(f"Loaded user sessions from {self.user_sessions_file}")
            logger.info(f"Loaded videos data from {self.videos_file}")
            
            return user_sessions, videos
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _get_bert_embedding(self, text: str):
        """Generate BERT embedding for text"""
        if not self.bert_available:
            # Return random embedding if BERT is not available
            return [random.random() for _ in range(768)]
            
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(text, 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  max_length=512, 
                                  padding=True).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as text representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating BERT embedding: {str(e)}")
            # Return random embedding as fallback
            return [random.random() for _ in range(768)]
    
    def calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate cosine similarity between two text descriptions using BERT embeddings"""
        if not self.bert_available or not ADVANCED_IMPORTS_AVAILABLE:
            # Return random similarity if BERT is not available
            return round(random.uniform(0.1, 0.9), 4)
            
        try:
            # Get embeddings
            emb1 = self._get_bert_embedding(desc1)
            emb2 = self._get_bert_embedding(desc2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            return round(float(similarity), 4)
        except Exception as e:
            logger.error(f"Error calculating description similarity: {str(e)}")
            # Return random similarity as fallback
            return round(random.uniform(0.1, 0.9), 4)
    
    def check_category_match(self, cat1: str, cat2: str) -> str:
        """Check if two categories match"""
        return "Y" if cat1 == cat2 else "N"
    
    def get_random_unwatched_video(self, watched_videos: List[str], all_videos: List[str]) -> str:
        """Get a random video that the user hasn't watched"""
        unwatched_videos = [v for v in all_videos if v not in watched_videos]
        if unwatched_videos:
            return random.choice(unwatched_videos)
        else:
            # If all videos have been watched (unlikely), return a random video
            return random.choice(all_videos)
    
    def process_data(self):
        """Process the data and create the preprocessed CSV file"""
        try:
            # Load data
            user_sessions, videos = self.load_data()
            
            # Create a dictionary for quick video lookup
            video_dict = {}
            all_video_ids = []
            
            # Process videos data - handle both pandas DataFrame and list of dicts
            if ADVANCED_IMPORTS_AVAILABLE and isinstance(videos, pd.DataFrame):
                for _, row in videos.iterrows():
                    video_id = row['video_id']
                    video_dict[video_id] = {
                        'description': row['description'],
                        'category': row['category']
                    }
                    all_video_ids.append(video_id)
            else:
                # Handle list of dictionaries from basic CSV reader
                for row in videos:
                    if isinstance(row, dict) and 'video_id' in row:
                        video_id = row['video_id']
                        video_dict[video_id] = {
                            'description': row['description'],
                            'category': row['category']
                        }
                        all_video_ids.append(video_id)
            
            # Prepare output data
            output_data = []
            
            # Process each user session
            total_sessions = len(user_sessions)
            
            # Handle different data formats
            if ADVANCED_IMPORTS_AVAILABLE and isinstance(user_sessions, pd.DataFrame):
                # Process pandas DataFrame
                for idx, session in user_sessions.iterrows():
                    user_id = session['user_id']
                    # Get all videos in this session (skip empty cells)
                    session_videos = [v for v in session[1:] if isinstance(v, str) and v]
                    
                    self._process_session_with_relevance(user_id, session_videos, video_dict, all_video_ids, output_data)
                    
                    # Log progress
                    if (idx + 1) % 100 == 0 or idx == len(user_sessions) - 1:
                        logger.info(f"Processed {idx + 1}/{len(user_sessions)} user sessions")
            else:
                # Process list of dictionaries
                for idx, session in enumerate(user_sessions):
                    if isinstance(session, dict) and 'user_id' in session:
                        user_id = session['user_id']
                        
                        # Get all videos in this session (skip empty cells)
                        session_videos = []
                        for key, value in session.items():
                            if key != 'user_id' and value and isinstance(value, str):
                                session_videos.append(value)
                        
                        self._process_session_with_relevance(user_id, session_videos, video_dict, all_video_ids, output_data)
                        
                    # Log progress
                    if (idx + 1) % 100 == 0 or idx == total_sessions - 1:
                        logger.info(f"Processed {idx + 1}/{total_sessions} user sessions")
            
            # Write to CSV
            with open(self.output_file, 'w', newline='') as csvfile:
                fieldnames = ['user', 'video1', 'video2', 'description_similarity', 'category_match', 'relevance_label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in output_data:
                    writer.writerow(row)
            
            logger.info(f"Created preprocessed data file with {len(output_data)} video pairs")
            logger.info(f"Output file: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _process_session_with_relevance(self, user_id, session_videos, video_dict, all_video_ids, output_data):
        """Process a single user session and add video pairs to output data with relevance labels"""
        if len(session_videos) < 1:
            return  # Skip sessions with no videos
        
        # Add relevance labels based on order in session (first=1, second=2, etc.)
        for i, video1 in enumerate(session_videos):
            # Skip if video is not in our video dictionary
            if video1 not in video_dict:
                continue
                
            # Relevance is based on position (1-indexed)
            relevance = i + 1
            
            # Create pairs with other videos in the session
            for j, video2 in enumerate(session_videos):
                if i == j or video2 not in video_dict:
                    continue  # Skip self-pairs and videos not in dictionary
                
                # Get video information
                video1_info = video_dict[video1]
                video2_info = video_dict[video2]
                
                # Calculate description similarity
                desc_similarity = self.calculate_description_similarity(
                    video1_info['description'], 
                    video2_info['description']
                )
                
                # Check category match
                category_match = self.check_category_match(
                    video1_info['category'],
                    video2_info['category']
                )
                
                # Add to output data with relevance label
                output_data.append({
                    'user': user_id,
                    'video1': video1,
                    'video2': video2,
                    'description_similarity': desc_similarity,
                    'category_match': category_match,
                    'relevance_label': relevance
                })
            
            # Add one unwatched video with relevance=0
            unwatched_video = self.get_random_unwatched_video(session_videos, all_video_ids)
            
            if unwatched_video and unwatched_video in video_dict:
                # Get video information
                video1_info = video_dict[video1]
                unwatched_info = video_dict[unwatched_video]
                
                # Calculate description similarity
                desc_similarity = self.calculate_description_similarity(
                    video1_info['description'], 
                    unwatched_info['description']
                )
                
                # Check category match
                category_match = self.check_category_match(
                    video1_info['category'],
                    unwatched_info['category']
                )
                
                # Add to output data with relevance=0
                output_data.append({
                    'user': user_id,
                    'video1': video1,
                    'video2': unwatched_video,
                    'description_similarity': desc_similarity,
                    'category_match': category_match,
                    'relevance_label': 0  # Unwatched video gets relevance 0
                })

def main():
    """Main function to preprocess data for learning to rank"""
    preprocessor = DataPreprocessor()
    preprocessor.process_data()

if __name__ == "__main__":
    main() 