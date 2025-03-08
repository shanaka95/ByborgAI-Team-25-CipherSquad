import os
import csv, llm
import pandas as pd
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Try to import the real recommender
    from recommender import VideoRecommender
    logger.info("Using real VideoRecommender")
except ImportError:
    # Fall back to the mock recommender if the real one is not available
    from mock_recommender import VideoRecommender
    logger.info("Using mock VideoRecommender")
from prompt import generate_preference_summary_prompt
from llm import generate_preference_summary

class RefinementProcessor:
    """
    A class to process and refine video recommendations by combining
    product descriptions and user search queries.
    """
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """
        Initialize the RefinementProcessor.
        
        Args:
            qdrant_host: Host address for Qdrant vector database
            qdrant_port: Port for Qdrant vector database
        """
        self.recommender = VideoRecommender(qdrant_host, qdrant_port)
        logger.info("Initialized RefinementProcessor")
        
    def get_user_search_queries(self, user_id: str, csv_path: str) -> List[str]:
        """
        Get all search queries for a specific user from the CSV file.
        
        Args:
            user_id: The ID of the user
            csv_path: Path to the user search queries CSV file
            
        Returns:
            List of search queries for the user
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Filter by user_id
            user_queries = df[df['user_id'] == user_id]['search_query'].tolist()
            
            logger.info(f"Found {len(user_queries)} search queries for user {user_id}")
            return user_queries
            
        except Exception as e:
            logger.error(f"Error retrieving search queries for user {user_id}: {str(e)}")
            return []
    
    def get_recommended_products(self, user_id: str, session_videos: List[str], max_recommendations: int = 10) -> List[str]:
        """
        Get recommended products for a user based on their session videos.
        
        Args:
            user_id: The ID of the user
            session_videos: List of videos the user has watched
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of recommended video IDs
        """
        try:
            # Get recommendations as a list
            recommendations = self.recommender.get_recommendations_as_list(
                user_id=user_id,
                session_videos=session_videos,
                max_recommendations=max_recommendations
            )
            
            logger.info(f"Retrieved {len(recommendations)} recommended products for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations for user {user_id}: {str(e)}")
            return []
    
    def get_product_descriptions(self, video_ids: List[str]) -> Dict[str, str]:
        """
        Get descriptions for a list of video IDs.
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            Dictionary mapping video IDs to their descriptions
        """
        descriptions = {}
        
        for video_id in video_ids:
            try:
                # Get video description from vector database
                video_data = self.recommender.get_video_description(video_id)
                
                if "error" in video_data:
                    logger.warning(f"Error retrieving description for video {video_id}: {video_data['error']}")
                    descriptions[video_id] = "No description available"
                else:
                    # Extract description from payload
                    description = video_data.get("description", "No description available")
                    descriptions[video_id] = description
                    
            except Exception as e:
                logger.error(f"Error processing description for video {video_id}: {str(e)}")
                descriptions[video_id] = "No description available"
        
        logger.info(f"Retrieved descriptions for {len(descriptions)} videos")
        return descriptions
    
    def combine_text(self, user_id: str, session_videos: List[str], queries_csv_path: str) -> str:
        """
        Combine product descriptions and user search queries into a single text.
        
        Args:
            user_id: The ID of the user
            session_videos: List of videos the user has watched
            queries_csv_path: Path to the user search queries CSV file
            
        Returns:
            Combined text of product descriptions and user search queries
        """
        # Get recommended products
        recommended_products = self.get_recommended_products(user_id, session_videos)
        
        if not recommended_products:
            logger.warning(f"No recommended products found for user {user_id}")
            return "No recommendations available."
        
        # Get product descriptions
        product_descriptions = self.get_product_descriptions(recommended_products)
        
        # Get user search queries
        user_queries = self.get_user_search_queries(user_id, queries_csv_path)
        
        # Combine descriptions
        combined_descriptions = "\n\n".join([
            f"Video {video_id}: {description}"
            for video_id, description in product_descriptions.items()
            if description != "No description available"
        ])
        
        # Combine search queries
        combined_queries = "\n".join([f"- {query}" for query in user_queries])
        
        # Create the final combined text
        combined_text = f"""
RECOMMENDED VIDEOS DESCRIPTIONS:
{combined_descriptions}

USER SEARCH QUERIES:
{combined_queries}
"""
        
        logger.info(f"Created combined text with {len(product_descriptions)} descriptions and {len(user_queries)} queries")
        return combined_text

def main():
    # Initialize the processor
    processor = RefinementProcessor()
    
    # Real user ID and session videos from user_sessions.csv
    user_id = "3babb493-6e30-49ed-91f5-1cefcc88f799"
    session_videos = ["scenecliptest00310.avi", "sceneclipautoautotrain00177.avi"]
    
    # Path to user search queries CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    queries_csv_path = os.path.join(current_dir, "user-sessions", "user_search_queries.csv")
    
    # Get combined text
    combined_text = processor.combine_text(user_id, session_videos, queries_csv_path)
    
    # Generate preference summary prompt
    preference_prompt = generate_preference_summary_prompt(combined_text)
    
    # Generate preference summary using LLM
    preference_summary = generate_preference_summary(preference_prompt)
    
    # Print the combined text
    print("\nCOMBINED TEXT:")
    print("=" * 80)
    print(combined_text)
    print("=" * 80)
    
    # Print the preference summary
    print("\nUSER PREFERENCE SUMMARY:")
    print("=" * 80)
    print(preference_summary)
    print("=" * 80)
    
    response = llm.generate_preference_summary(preference_prompt)

    return response

if __name__ == "__main__":
    main()
