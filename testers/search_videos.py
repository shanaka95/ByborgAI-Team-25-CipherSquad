import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
import json
from pathlib import Path
import numpy as np

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

def search_videos_by_text(
    query_text: str,
    collection_name: str = "videos",
    clip_model: str = "clip-ViT-B-16",
    limit: int = 10,
    threshold: float = 0.2,
    with_video_grouping: bool = True
) -> List[Dict[str, Any]]:
    """
    Search video frames using a text query.
    
    Args:
        query_text: Text to search for
        collection_name: Qdrant collection name
        clip_model: CLIP model to use
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        with_video_grouping: Group results by video
        
    Returns:
        List of search results
    """
    # Initialize CLIP encoder
    clip_encoder = ClipEncoder(model_name=clip_model)
    
    # Generate text embedding
    text_embedding = clip_encoder.encode_texts(query_text)
    
    # Initialize Qdrant manager
    qdrant = QdrantManager(
        collection_name=collection_name,
        vector_size=clip_encoder.embedding_dim
    )
    
    # Search for similar vectors
    results = qdrant.search(
        query_vector=text_embedding.tolist()[0],
        limit=limit * (3 if with_video_grouping else 1),  # Get more results if grouping
        score_threshold=threshold
    )
    
    if not results:
        logger.info(f"No results found for query: {query_text}")
        return []
    
    logger.info(f"Found {len(results)} results for query: {query_text}")
    
    if with_video_grouping:
        # Group results by video
        video_groups = {}
        for result in results:
            video_name = result["payload"]["video_name"]
            if video_name not in video_groups:
                video_groups[video_name] = []
            
            # Keep only first 3 frames per video to avoid redundancy
            if len(video_groups[video_name]) < 3:
                video_groups[video_name].append(result)
        
        # Take top matches from each video
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
        results = grouped_results[:limit]
    
    return results

def search_videos_by_image(
    image_path: str,
    collection_name: str = "videos",
    clip_model: str = "clip-ViT-B-16",
    limit: int = 10,
    threshold: float = 0.2,
    with_video_grouping: bool = True
) -> List[Dict[str, Any]]:
    """
    Search video frames using an image.
    
    Args:
        image_path: Path to image file
        collection_name: Qdrant collection name
        clip_model: CLIP model to use
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        with_video_grouping: Group results by video
        
    Returns:
        List of search results
    """
    # Initialize CLIP encoder
    clip_encoder = ClipEncoder(model_name=clip_model)
    
    # Generate image embedding
    image_embedding = clip_encoder.encode_images(image_path)
    
    # Initialize Qdrant manager
    qdrant = QdrantManager(
        collection_name=collection_name,
        vector_size=clip_encoder.embedding_dim
    )
    
    # Search for similar vectors
    results = qdrant.search(
        query_vector=image_embedding.tolist()[0],
        limit=limit * (3 if with_video_grouping else 1),  # Get more results if grouping
        score_threshold=threshold
    )
    
    if not results:
        logger.info(f"No results found for image: {image_path}")
        return []
    
    logger.info(f"Found {len(results)} results for image: {image_path}")
    
    if with_video_grouping:
        # Group results by video (same as in text search)
        video_groups = {}
        for result in results:
            video_name = result["payload"]["video_name"]
            if video_name not in video_groups:
                video_groups[video_name] = []
            
            if len(video_groups[video_name]) < 3:
                video_groups[video_name].append(result)
        
        grouped_results = []
        for video_name, video_results in video_groups.items():
            video_results.sort(key=lambda x: x["score"], reverse=True)
            best_result = video_results[0]
            best_result["total_matches"] = len(video_results)
            best_result["avg_score"] = sum(r["score"] for r in video_results) / len(video_results)
            grouped_results.append(best_result)
        
        grouped_results.sort(key=lambda x: x["score"], reverse=True)
        results = grouped_results[:limit]
    
    return results

def print_results(results):
    """Print search results in a formatted way."""
    if results:
        logger.info(f"Top {len(results)} results:")
        for i, result in enumerate(results):
            video_name = result["payload"]["video_name"]
            timestamp = result["payload"]["timestamp"]
            score = result["score"]
            logger.info(f"{i+1}. Video: {video_name}, Time: {timestamp:.2f}s, Score: {score:.4f}")
            
            # If grouped, show total matches
            if "total_matches" in result:
                logger.info(f"   Total frames matched: {result['total_matches']}, Avg score: {result['avg_score']:.4f}")
    else:
        logger.info("No results found.")

def save_results_to_file(results, output_file):
    """Save results to a JSON file."""
    if not results:
        return
        
    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {
            "id": result["id"],
            "score": float(result["score"]),
            "payload": result["payload"]
        }
        if "total_matches" in result:
            clean_result["total_matches"] = result["total_matches"]
            clean_result["avg_score"] = float(result["avg_score"])
        clean_results.append(clean_result)
        
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

# Configuration variables instead of command-line arguments
if __name__ == "__main__":
    # Predefined variables for the script
    QUERY_TEXT = "person walking on beach"  # Text query to search for
    QUERY_IMAGE = None  # Image file to search with (None to use text query)
    COLLECTION_NAME = "videos"  # Collection name
    CLIP_MODEL = "clip-ViT-B-16"  # CLIP model to use
    RESULT_LIMIT = 10  # Maximum number of results
    SIMILARITY_THRESHOLD = 0.2  # Minimum similarity threshold
    GROUP_BY_VIDEO = True  # Group results by video
    OUTPUT_FILE = None  # Output file for results (JSON), None for no file output
    
    # Search based on provided query type
    if QUERY_IMAGE:
        results = search_videos_by_image(
            image_path=QUERY_IMAGE,
            collection_name=COLLECTION_NAME,
            clip_model=CLIP_MODEL,
            limit=RESULT_LIMIT,
            threshold=SIMILARITY_THRESHOLD,
            with_video_grouping=GROUP_BY_VIDEO
        )
    else:
        results = search_videos_by_text(
            query_text=QUERY_TEXT,
            collection_name=COLLECTION_NAME,
            clip_model=CLIP_MODEL,
            limit=RESULT_LIMIT,
            threshold=SIMILARITY_THRESHOLD,
            with_video_grouping=GROUP_BY_VIDEO
        )
    
    # Print results
    print_results(results)
    
    # Save results to file if specified
    if OUTPUT_FILE and results:
        save_results_to_file(results, OUTPUT_FILE) 