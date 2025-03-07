import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import from ByborgAI
sys.path.append(str(Path(__file__).parent.parent))
from Qdrant import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_qdrant_for_videos(
    collection_name: str = "videos",
    vector_size: int = 512,  # Default for CLIP ViT-B/16
    url: str = None,
    api_key: str = None,
    host: str = "localhost",
    port: int = 6333,
    recreate: bool = False,
    distance: str = "cosine",
    init_file_marker: str = ".qdrant-videos-initialized",
    on_disk_payload: bool = True
) -> bool:
    """
    Initialize a Qdrant collection for video frame embeddings.
    
    Args:
        collection_name: Name for the Qdrant collection
        vector_size: Size of the embedding vectors
        url: URL for cloud Qdrant (if used)
        api_key: API key for cloud Qdrant (if used)
        host: Host for local Qdrant
        port: Port for local Qdrant
        recreate: Force recreation even if already initialized
        distance: Distance metric to use ("cosine", "euclid", "dot")
        init_file_marker: File to create as an initialization marker
        on_disk_payload: Whether to store payload on disk instead of RAM
        
    Returns:
        bool: Success status
    """
    # Check if already initialized (using a marker file)
    marker_path = Path(__file__).parent / init_file_marker
    if marker_path.exists() and not recreate:
        logger.info(f"Qdrant videos collection already initialized (marker file found at {marker_path})")
        return True
    
    logger.info(f"Initializing Qdrant collection '{collection_name}' for video processing")
    
    # Create QdrantManager instance
    qdrant = QdrantManager(
        collection_name=collection_name,
        vector_size=vector_size,
        url=url,
        api_key=api_key,
        host=host,
        port=port
    )
    
    # Configure HNSW index for optimal similarity search
    hnsw_config = {
        "m": 16,  # Number of connections per layer (higher = better recall, more memory)
        "ef_construct": 100,  # Size of the dynamic candidate list (higher = better recall, slower build)
        "full_scan_threshold": 10000  # Number of points for full scan vs. indexed search
    }
    
    # Configure optimizers
    optimizers_config = {
        "default_segment_number": 2,  # Number of segments in the collection
        "indexing_threshold": 20000,  # Number of vectors to trigger indexing
        "memmap_threshold": 50000,    # Number of vectors to trigger memmap
    }
    
    # Create the collection with optimal settings
    success = qdrant.create_collection(
        distance=distance,
        on_disk_payload=on_disk_payload,
        hnsw_config=hnsw_config,
        optimizers_config=optimizers_config,
        force_recreate=recreate
    )
    
    if success:
        # Create initialization marker file
        with open(marker_path, 'w') as f:
            f.write(f"Qdrant collection '{collection_name}' initialized for video processing")
        
        logger.info(f"Successfully initialized Qdrant collection '{collection_name}' for video processing")
        logger.info(f"Created marker file at {marker_path}")
        
        # Get and log collection info
        collection_info = qdrant.get_collection_info()
        logger.info(f"Collection info: {collection_info}")
        
        return True
    else:
        logger.error(f"Failed to initialize Qdrant collection '{collection_name}'")
        return False

def create_payload_index(
    collection_name: str = "videos",
    url: str = None,
    api_key: str = None,
    host: str = "localhost", 
    port: int = 6333
):
    """
    Create payload indexes for efficient filtering.
    
    Args:
        collection_name: Name of the Qdrant collection
        url: URL for cloud Qdrant (if used)
        api_key: API key for cloud Qdrant (if used)
        host: Host for local Qdrant
        port: Port for local Qdrant
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    
    # Connect to Qdrant
    if url and api_key:
        client = QdrantClient(url=url, api_key=api_key)
    else:
        client = QdrantClient(host=host, port=port)
    
    # Create index for video_name field (for quick filtering by video)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="video_name",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    # Create index for timestamp field (for quick filtering by timestamp)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="timestamp",
        field_schema=models.PayloadSchemaType.FLOAT
    )
    
    logger.info(f"Created payload indexes for collection '{collection_name}'")

# Configuration variables instead of command-line arguments
if __name__ == "__main__":
    # Predefined variables for the script
    COLLECTION_NAME = "videos"  # Collection name
    VECTOR_SIZE = 512  # Vector size for embeddings (512 for CLIP ViT-B/16)
    HOST = "localhost"  # Qdrant host
    PORT = 6333  # Qdrant port
    RECREATE = False  # Force recreation
    CLOUD_URL = None  # Qdrant Cloud URL
    CLOUD_API_KEY = None  # Qdrant Cloud API key
    CREATE_INDEXES = True  # Create payload indexes
    
    # Initialize collection
    success = initialize_qdrant_for_videos(
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
        url=CLOUD_URL,
        api_key=CLOUD_API_KEY,
        host=HOST,
        port=PORT,
        recreate=RECREATE
    )
    
    # Optionally create payload indexes
    if success and CREATE_INDEXES:
        create_payload_index(
            collection_name=COLLECTION_NAME,
            url=CLOUD_URL,
            api_key=CLOUD_API_KEY,
            host=HOST,
            port=PORT
        ) 