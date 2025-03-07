import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add parent directory to path to import from ByborgAI
sys.path.append(str(Path(__file__).parent.parent))
from Qdrant import QdrantManager
from Clip import ClipEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_video_frames(
    query_text: str,
    num_results: int = 5,
    collection_name: str = "videos",
    clip_model: str = "clip-ViT-B-16",
    threshold: float = 0.2,
    group_by_video: bool = True
) -> List[Dict[str, Any]]:
    """
    Find video frames matching a text query.
    
    Args:
        query_text: Text description to search for
        num_results: Number of results to return
        collection_name: Qdrant collection name
        clip_model: CLIP model to use
        threshold: Similarity threshold (0-1)
        group_by_video: Group results by video
        
    Returns:
        List of matching video frames with metadata
    """
    logger.info(f"Searching for: '{query_text}'")
    
    # Initialize CLIP encoder
    clip_encoder = ClipEncoder(model_name=clip_model)
    
    # Generate text embedding
    text_embedding = clip_encoder.encode_texts(query_text)
    
    if text_embedding is None or len(text_embedding) == 0:
        logger.error("Failed to generate text embedding")
        return []
    
    # Initialize Qdrant manager
    qdrant = QdrantManager(
        collection_name=collection_name,
        vector_size=clip_encoder.embedding_dim
    )
    
    # Calculate multiplier for getting more results if grouping
    limit_multiplier = 3 if group_by_video else 1
    
    # Search for similar vectors
    results = qdrant.search(
        query_vector=text_embedding.tolist()[0],
        limit=num_results * limit_multiplier,
        score_threshold=threshold
    )
    
    if not results:
        logger.info(f"No matching video frames found for: '{query_text}'")
        return []
    
    logger.info(f"Found {len(results)} matching frames")
    
    # Group results by video if requested
    if group_by_video:
        video_groups = {}
        for result in results:
            video_name = result["payload"]["video_name"]
            if video_name not in video_groups:
                video_groups[video_name] = []
            
            # Keep only first 3 frames per video
            if len(video_groups[video_name]) < 3:
                video_groups[video_name].append(result)
        
        # Get best result from each video
        grouped_results = []
        for video_name, video_results in video_groups.items():
            # Sort by score (highest first)
            video_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Get best frame
            best_result = video_results[0]
            
            # Add video info
            best_result["total_matches"] = len(video_results)
            best_result["avg_score"] = sum(r["score"] for r in video_results) / len(video_results)
            
            grouped_results.append(best_result)
        
        # Sort groups by best score
        grouped_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested number
        results = grouped_results[:num_results]
    
    # Format results for easy consumption
    formatted_results = []
    for idx, result in enumerate(results):
        formatted_result = {
            "rank": idx + 1,
            "video_name": result["payload"]["video_name"],
            "video_path": result["payload"]["video_path"],
            "timestamp": result["payload"]["timestamp"],
            "score": result["score"]
        }
        
        # Add grouping info if available
        if "total_matches" in result:
            formatted_result["total_matches"] = result["total_matches"]
            formatted_result["avg_score"] = result["avg_score"]
            
        formatted_results.append(formatted_result)
    
    return formatted_results

def print_results(results: List[Dict[str, Any]]) -> None:
    """
    Print search results in a readable format.
    
    Args:
        results: List of search results
    """
    if not results:
        print("No results found.")
        return
        
    print(f"\nFound {len(results)} matching videos:\n" + "-" * 50)
    
    for result in results:
        print(f"{result['rank']}. {result['video_name']}")
        print(f"   Time: {result['timestamp']:.2f}s")
        print(f"   Score: {result['score']:.4f}")
        
        if "total_matches" in result:
            print(f"   Total frames matched: {result['total_matches']}")
            print(f"   Average score: {result['avg_score']:.4f}")
        
        print(f"   Path: {result['video_path']}")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Simple search interface
    QUERY = "a ceiling fan spinning"  # Example query - change this
    NUM_RESULTS = 5  # Number of results to return
    COLLECTION = "videos"  # Qdrant collection name
    
    # Perform search
    results = find_video_frames(
        query_text=QUERY,
        num_results=NUM_RESULTS,
        collection_name=COLLECTION
    )
    
    # Print results
    print_results(results) 