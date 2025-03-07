import os
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRecommender:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize the video recommender with BERT model and Qdrant client"""
        # Connect to Qdrant
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.descriptions_collection = "video_descriptions"
        
        # Initialize BERT model and tokenizer
        logger.info("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"BERT model loaded successfully (using {self.device})")
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embedding for text"""
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
            raise
    
    def get_video_description(self, video_name: str) -> Dict[str, Any]:
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
                return {"error": f"No description found for {video_name}"}
            
            return points[0].payload
            
        except Exception as e:
            logger.error(f"Error retrieving description for {video_name}: {str(e)}")
            return {"error": str(e)}
    
    def find_similar_videos(self, description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find videos with similar descriptions using BERT embeddings"""
        try:
            # Generate embedding for the description
            query_embedding = self._get_bert_embedding(description)
            
            # Search for similar descriptions
            search_results = self.client.search(
                collection_name=self.descriptions_collection,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            # Format results
            similar_videos = []
            for result in search_results:
                similar_videos.append({
                    "video_id": result.payload.get("video_id", "unknown"),
                    "description": result.payload.get("description", "No description available"),
                    "similarity_score": result.score
                })
                
            return similar_videos
            
        except Exception as e:
            logger.error(f"Error searching for similar videos: {str(e)}")
            return []
    
    def get_recommendations_for_session(self, session_videos: List[str], recommendations_per_video: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get recommendations based on a list of previously watched videos"""
        recommendations = {}
        
        for video_name in session_videos:
            # Get video description
            video_info = self.get_video_description(video_name)
            
            if "error" in video_info:
                logger.warning(f"Skipping recommendations for {video_name}: {video_info['error']}")
                recommendations[video_name] = []
                continue
            
            # Get description text
            description = video_info.get("description", "")
            if not description:
                logger.warning(f"No description available for {video_name}")
                recommendations[video_name] = []
                continue
            
            # Find similar videos
            similar_videos = self.find_similar_videos(description, limit=recommendations_per_video)
            
            # Filter out the original video from recommendations
            filtered_recommendations = [
                video for video in similar_videos 
                if video["video_id"] != video_name
            ]
            
            recommendations[video_name] = filtered_recommendations
            
        return recommendations

def main():
    # Set user ID
    user_id = "d5d292d0-ba64-4771-86fe-32a6a44716f3"
    
    # Set previously visited sessions
    previous_sessions = [
        "sceneclipautoautotrain00301.avi",
        "sceneclipautoautotrain00213.avi",
        "scenecliptest00006.avi"
    ]
    
    # Initialize recommender
    recommender = VideoRecommender()
    
    # Get recommendations for each previously visited session
    recommendations = recommender.get_recommendations_for_session(previous_sessions)
    
    # Print recommendations
    print(f"Recommendations for user {user_id}:")
    print("=" * 50)
    
    for video_name, similar_videos in recommendations.items():
        print(f"\nBased on your interest in: {video_name}")
        
        if not similar_videos:
            print("  No recommendations found.")
            continue
            
        print("  You might also like:")
        for i, video in enumerate(similar_videos, 1):
            print(f"  {i}. {video['video_id']} (Similarity: {video['similarity_score']:.2f})")
            print(f"     Description: {video['description'][:100]}...")

if __name__ == "__main__":
    main()
