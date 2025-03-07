import os
import cv2
import time
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from Clip import ClipEncoder
from Qdrant import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_video_files(directory: str, extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> List[str]:
    """
    List all video files in a directory with specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of video file extensions to include
        
    Returns:
        List of video file paths
    """
    video_files = []
    for ext in extensions:
        video_files.extend(list(Path(directory).glob(f"**/*{ext}")))
    
    return [str(path) for path in video_files]

def extract_frames(
    video_path: str, 
    fps: int = 1,
    max_frames: Optional[int] = None
) -> List[Tuple[np.ndarray, float]]:
    """
    Extract frames from a video at a specified rate.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        List of tuples containing (frame, timestamp)
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file {video_path}")
            return frames
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Duration: {duration:.2f}s, FPS: {video_fps}, Total frames: {total_frames}")
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        # Extract frames at specified intervals
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Calculate timestamp
                    timestamp = frame_count / video_fps
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add frame to list
                    frames.append((frame_rgb, timestamp))
                    extracted_count += 1
                    
                    # Check if we've reached the maximum number of frames
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
    
    return frames

def is_video_already_processed(qdrant: QdrantManager, video_path: str) -> bool:
    """
    Check if a video has already been processed and stored in Qdrant.
    
    Args:
        qdrant: QdrantManager instance
        video_path: Path to the video file
        
    Returns:
        True if video has already been processed, False otherwise
    """
    # Create a filter to search for the video path in payloads
    filter_conditions = {
        "must": [
            {
                "key": "video_path",
                "match": {"value": video_path}
            }
        ]
    }
    
    # Count points matching the filter
    count = qdrant.count_points(filter_conditions=filter_conditions)
    
    return count > 0

def process_video_directory(
    data_dir: str,
    clip_model: str = "clip-ViT-B-16",
    qdrant_collection: str = "video_frames",
    fps: int = 1,
    max_frames_per_video: Optional[int] = None,
    batch_size: int = 32,
    recreate_collection: bool = False,
    skip_existing: bool = True
) -> None:
    """
    Process all videos in a directory, extract frames, and store embeddings in Qdrant.
    
    Args:
        data_dir: Directory containing video files
        clip_model: CLIP model to use
        qdrant_collection: Qdrant collection name
        fps: Frames per second to extract
        max_frames_per_video: Maximum frames to extract per video
        batch_size: Batch size for processing
        recreate_collection: Whether to recreate the Qdrant collection
        skip_existing: Whether to skip videos that have already been processed
    """
    # Initialize CLIP encoder
    logger.info(f"Initializing CLIP encoder with model {clip_model}")
    clip_encoder = ClipEncoder(model_name=clip_model)
    
    # Initialize Qdrant manager
    logger.info(f"Initializing Qdrant manager with collection {qdrant_collection}")
    qdrant = QdrantManager(
        collection_name=qdrant_collection,
        vector_size=clip_encoder.embedding_dim
    )
    
    # Create or validate collection
    qdrant.create_collection(force_recreate=recreate_collection)
    
    # List video files
    logger.info(f"Scanning directory: {data_dir}")
    video_files = list_video_files(data_dir)
    logger.info(f"Found {len(video_files)} video files")
    
    # Count for statistics
    videos_processed = 0
    videos_skipped = 0
    frames_processed = 0
    
    # Process each video
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        
        # Check if video has already been processed
        if skip_existing and is_video_already_processed(qdrant, video_path):
            logger.info(f"Video already processed, skipping: {video_name}")
            videos_skipped += 1
            continue
            
        logger.info(f"Processing video: {video_name}")
        
        # Extract frames
        frames = extract_frames(
            video_path=video_path,
            fps=fps,
            max_frames=max_frames_per_video
        )
        
        if not frames:
            logger.warning(f"No frames extracted from {video_name}, skipping")
            continue
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Convert frames to PIL images
            pil_images = [Image.fromarray(frame) for frame, _ in batch_frames]
            timestamps = [ts for _, ts in batch_frames]
            
            # Generate embeddings
            embeddings = clip_encoder.encode_images(pil_images, batch_size=batch_size)
            
            # Prepare point IDs and payloads
            ids = [str(uuid.uuid4()) for _ in range(len(batch_frames))]
            payloads = [
                {
                    "video_path": video_path,
                    "video_name": video_name,
                    "timestamp": ts,
                    "frame_index": i + idx,
                    "processed_at": time.time()
                }
                for idx, ts in enumerate(timestamps)
            ]
            
            # Store embeddings in Qdrant
            qdrant.add_points(
                ids=ids,
                vectors=embeddings.tolist(),
                payloads=payloads
            )
            
            frames_processed += len(batch_frames)
            logger.info(f"Added {len(batch_frames)} embeddings from {video_name} to Qdrant")
        
        videos_processed += 1
    
    # Get final stats
    collection_info = qdrant.get_collection_info()
    logger.info(f"Processing complete:")
    logger.info(f"- Videos processed: {videos_processed}")
    logger.info(f"- Videos skipped (already processed): {videos_skipped}")
    logger.info(f"- Total frames processed: {frames_processed}")
    logger.info(f"- Collection stats: {collection_info}")


# Configuration variables instead of command-line arguments
if __name__ == "__main__":
    # Predefined variables for the script
    DATA_DIR = "data"  # Directory containing video files
    CLIP_MODEL = "clip-ViT-B-16"  # CLIP model to use
    QDRANT_COLLECTION = "videos"  # Qdrant collection name
    FPS = 1  # Frames per second to extract
    MAX_FRAMES = None  # Maximum frames to extract per video (None for all)
    BATCH_SIZE = 32  # Batch size for processing
    RECREATE_COLLECTION = False  # Whether to recreate the Qdrant collection
    SKIP_EXISTING = True  # Whether to skip videos that have already been processed
    
    # Run the processing function with predefined variables
    process_video_directory(
        data_dir=DATA_DIR,
        clip_model=CLIP_MODEL,
        qdrant_collection=QDRANT_COLLECTION,
        fps=FPS,
        max_frames_per_video=MAX_FRAMES,
        batch_size=BATCH_SIZE,
        recreate_collection=RECREATE_COLLECTION,
        skip_existing=SKIP_EXISTING
    ) 