from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import logging
import uuid
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VideoDescriptionManager:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.videos_collection = "videos"
        self.descriptions_collection = "video_descriptions"
        
    def setup_collections(self):
        """Set up both videos and descriptions collections with proper schema"""
        try:
            # Set up descriptions collection
            self._ensure_descriptions_collection()
            # Set up videos collection (if not exists)
            self._ensure_videos_collection()
            logger.info("Collections setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up collections: {str(e)}")
            raise

    def _ensure_descriptions_collection(self):
        """Create video descriptions collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.descriptions_collection for collection in collections)
            
            if not exists:
                # Create collection for descriptions
                self.client.create_collection(
                    collection_name=self.descriptions_collection,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.descriptions_collection}")
                
                # Create payload indices
                self._create_description_indices()
        except Exception as e:
            logger.error(f"Error creating descriptions collection: {str(e)}")
            raise
                
    def _ensure_videos_collection(self):
        """Create videos collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.videos_collection for collection in collections)
            
            if not exists:
                # Create collection for videos
                self.client.create_collection(
                    collection_name=self.videos_collection,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.videos_collection}")
                
                # Create payload indices for videos
                self.client.create_payload_index(
                    collection_name=self.videos_collection,
                    field_name="videos",
                    field_type="keyword"
                )
        except Exception as e:
            logger.error(f"Error creating videos collection: {str(e)}")
            raise

    def _create_description_indices(self):
        """Create necessary payload indices for descriptions collection"""
        indices = [
            ("video_id", "keyword"),
            ("description", "text"),
            ("timestamp", "float"),
            ("model_used", "keyword")
        ]
        
        for field_name, field_type in indices:
            self.client.create_payload_index(
                collection_name=self.descriptions_collection,
                field_name=field_name,
                field_type=field_type
            )
        logger.info("Created payload indices for descriptions collection")

    def save_description(self, video_name: str, description: Dict) -> str:
        """
        Save a video description to the descriptions collection and link it to the videos collection
        Returns the ID of the created description
        """
        try:
            # Generate unique ID for the description
            description_id = str(uuid.uuid4())
            
            # Create the description payload
            description_payload = {
                "video_id": video_name,
                "description": description["description"],
                "model_used": description["model_used"],
                "num_frames": description["num_frames_analyzed"],
                "timestamp": time.time()
            }
            
            # Save description
            self.client.upsert(
                collection_name=self.descriptions_collection,
                points=models.Batch(
                    ids=[description_id],
                    vectors=[[0.0] * 512],  # Placeholder vector
                    payloads=[description_payload]
                )
            )
            
            # Update video record with description reference
            self._update_video_description_ref(video_name, description_id)
            
            logger.info(f"Saved description for video: {video_name}")
            return description_id
            
        except Exception as e:
            logger.error(f"Error saving description for {video_name}: {str(e)}")
            raise

    def _update_video_description_ref(self, video_name: str, description_id: str):
        """Update the video record with reference to its description"""
        try:
            # Find the video record
            search_results = self.client.scroll(
                collection_name=self.videos_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="videos",
                            match=models.MatchValue(value=video_name)
                        )
                    ]
                ),
                limit=1
            )
            
            points = search_results[0]
            
            if points:
                # Update existing video record
                self.client.set_payload(
                    collection_name=self.videos_collection,
                    payload={"description_id": description_id},
                    points=[points[0].id]
                )
            else:
                # Create new video record
                video_id = str(uuid.uuid4())
                self.client.upsert(
                    collection_name=self.videos_collection,
                    points=models.Batch(
                        ids=[video_id],
                        vectors=[[0.0] * 512],  # Placeholder vector
                        payloads=[{
                            "videos": video_name,
                            "description_id": description_id
                        }]
                    )
                )
                
        except Exception as e:
            logger.error(f"Error updating video reference for {video_name}: {str(e)}")
            raise

    def get_video_description(self, video_name: str) -> Optional[Dict]:
        """Retrieve the latest description for a video"""
        try:
            # Find the video's description reference
            video_results = self.client.scroll(
                collection_name=self.videos_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="videos",
                            match=models.MatchValue(value=video_name)
                        )
                    ]
                ),
                limit=1
            )
            
            points = video_results[0]
            if not points:
                logger.warning(f"No video record found for: {video_name}")
                return None
                
            description_id = points[0].payload.get("description_id")
            if not description_id:
                logger.warning(f"No description reference found for video: {video_name}")
                return None
                
            # Get the description
            description_results = self.client.scroll(
                collection_name=self.descriptions_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=description_id)
                        )
                    ]
                ),
                limit=1
            )
            
            desc_points = description_results[0]
            if not desc_points:
                logger.warning(f"Description not found for ID: {description_id}")
                return None
                
            return desc_points[0].payload
            
        except Exception as e:
            logger.error(f"Error retrieving description for {video_name}: {str(e)}")
            raise

def main():
    """Set up collections and test functionality"""
    try:
        # Initialize manager
        manager = VideoDescriptionManager()
        
        # Set up collections
        manager.setup_collections()
        
        print("Video description collections have been set up successfully!")
        print("\nCollection Structure:")
        print("1. videos collection:")
        print("   - videos (keyword index)")
        print("   - description_id (reference to description)")
        print("\n2. video_descriptions collection:")
        print("   - video_id (keyword index)")
        print("   - description (text index)")
        print("   - timestamp (float index)")
        print("   - model_used (keyword index)")
        print("   - num_frames (stored in payload)")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 