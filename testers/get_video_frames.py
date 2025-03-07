from qdrant_client import QdrantClient, models
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def get_all_frames(video_name: str) -> List[Dict]:
    """
    Retrieve all frames and their metadata for a specific video from Qdrant.
    Returns a list of dictionaries containing frame data sorted by frame number.
    """
    try:
        # Initialize Qdrant client
        client = QdrantClient(url="http://localhost:6333")
        
        # Scroll through all frames for the given video
        results = client.scroll(
            collection_name="videos",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_name",
                        match=models.MatchValue(value=video_name),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=True
        )
        
        # Extract frame data
        frame_data = []
        for record in results[0]:
            frame_info = {
                "frame_index": record.payload["frame_index"],
                "embedding": np.array(record.vector),
                **record.payload  # Include all other payload fields
            }
            frame_data.append(frame_info)
        
        # Sort by frame number
        frame_data.sort(key=lambda x: x["frame_index"])
        
        logger.info(f"Retrieved {len(frame_data)} frames for video: {video_name}")
        return frame_data
    
    except Exception as e:
        logger.error(f"Error retrieving frames for video {video_name}: {e}")
        raise

def main():
    """
    Example usage of the get_all_frames function
    """
    video_name = "sceneclipautoautotrain00012.avi"
    try:
        frames = get_all_frames(video_name)
        print(f"Found {len(frames)} frames for video: {video_name}")
        print("\nFirst frame data:")
        print(f"Frame number: {frames[0]['frame_index']}")
        print(f"Embedding shape: {frames[0]['embedding'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 