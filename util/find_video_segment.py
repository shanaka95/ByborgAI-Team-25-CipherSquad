from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def get_frame_embeddings(client: QdrantClient, video_name: str) -> List[Tuple[np.ndarray, int]]:
    """
    Retrieve embeddings for all frames of a specific video from Qdrant.
    Returns list of (embedding, frame_index) tuples.
    """
    try:
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
        
        # Extract embeddings and frame numbers
        frame_data = []
        for record in results[0]:
            frame_data.append((np.array(record.vector), record.payload["frame_index"]))
        
        # Sort by frame index
        frame_data.sort(key=lambda x: x[1])
        return frame_data
    
    except Exception as e:
        logger.error(f"Error retrieving frame embeddings: {e}")
        raise

def find_best_matching_segment(text: str, video_name: str, window_size: int = 5) -> Tuple[int, int]:
    """
    Find the video segment that best matches the given text.
    Returns (start_frame, end_frame) where the segment is exactly window_size frames long,
    starting from the best matching frame.
    """
    try:
        # Initialize clients
        client = QdrantClient(url="http://localhost:6333")
        model = SentenceTransformer('clip-ViT-B-16')
        
        # Generate text embedding
        text_embedding = model.encode(text)
        
        # Get all frame embeddings
        frame_embeddings = get_frame_embeddings(client, video_name)
        
        if len(frame_embeddings) < window_size:
            raise ValueError(f"Video has fewer frames ({len(frame_embeddings)}) than window size ({window_size})")
        
        # Calculate sliding window averages
        min_avg_distance = float('inf')
        best_start_idx = 0
        
        for i in range(len(frame_embeddings) - window_size + 1):
            window_distances = []
            for j in range(window_size):
                frame_embedding = frame_embeddings[i + j][0]
                # Calculate cosine distance (1 - cosine similarity)
                distance = 1 - np.dot(text_embedding, frame_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(frame_embedding)
                )
                window_distances.append(distance)
            
            avg_distance = np.mean(window_distances)
            
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_start_idx = i
        
        # Get the actual frame indices
        start_frame = frame_embeddings[best_start_idx][1]
        end_frame = start_frame + window_size   # Simply add window_size-1 to get exact number of frames
        
        return start_frame, end_frame
    
    except Exception as e:
        logger.error(f"Error finding best matching segment: {e}")
        raise

def main():
    """
    Example usage of the find_best_matching_segment function
    """
    video_name = "sceneclipautoautotrain00132.avi"
    text = "pillow"
    window_size = 5
    
    try:
        start_frame, end_frame = find_best_matching_segment(text, video_name, window_size)
        print(f"Best matching segment for text '{text}':")
        print(f"Video: {video_name}")
        print(f"Frames {start_frame} to {end_frame}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 