from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import logging
import uuid
import time
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VideoDescriptionManager:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant and BERT model"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.videos_collection = "videos"
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
                # Create collection for descriptions with BERT embedding dimension
                self.client.create_collection(
                    collection_name=self.descriptions_collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # BERT base outputs 768-dim vectors
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
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Match BERT dimension
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
            
            # Generate BERT embedding for the description
            logger.info(f"Generating embedding for video: {video_name}")
            embedding = self._get_bert_embedding(description["description"])
            
            # Create the description payload
            description_payload = {
                "video_id": video_name,
                "description": description["description"],
                "model_used": description["model_used"],
                "num_frames": description["num_frames_analyzed"],
                "timestamp": time.time()
            }
            
            # Save description with embedding
            self.client.upsert(
                collection_name=self.descriptions_collection,
                points=models.Batch(
                    ids=[description_id],
                    vectors=[embedding.tolist()],
                    payloads=[description_payload]
                )
            )
            
            # Update video record with description reference and embedding
            self._update_video_description_ref(video_name, description_id, embedding)
            
            logger.info(f"Saved description and embedding for video: {video_name}")
            return description_id
            
        except Exception as e:
            logger.error(f"Error saving description for {video_name}: {str(e)}")
            raise

    def _update_video_description_ref(self, video_name: str, description_id: str, embedding: np.ndarray):
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
                        vectors=[embedding.tolist()],  # Use same embedding as description
                        payloads=[{
                            "videos": video_name,
                            "description_id": description_id
                        }]
                    )
                )
                
        except Exception as e:
            logger.error(f"Error updating video reference for {video_name}: {str(e)}")
            raise

    def find_similar_videos(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Find videos with similar descriptions using BERT embeddings"""
        try:
            # Generate embedding for query text
            query_embedding = self._get_bert_embedding(query_text)
            
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
                    "video_id": result.payload["video_id"],
                    "description": result.payload["description"],
                    "similarity_score": result.score
                })
                
            return similar_videos
            
        except Exception as e:
            logger.error(f"Error searching for similar videos: {str(e)}")
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
        print("   - vector (768-dim BERT embedding)")
        print("\n2. video_descriptions collection:")
        print("   - video_id (keyword index)")
        print("   - description (text index)")
        print("   - timestamp (float index)")
        print("   - model_used (keyword index)")
        print("   - vector (768-dim BERT embedding)")
        print("   - num_frames (stored in payload)")
        
        # Test semantic search functionality
        test_query = "people walking in a city"
        print(f"\nTesting semantic search with query: '{test_query}'")
        similar_videos = manager.find_similar_videos(test_query, limit=3)
        
        print("\nSimilar videos found:")
        for video in similar_videos:
            print(f"\nVideo: {video['video_id']}")
            print(f"Similarity score: {video['similarity_score']:.3f}")
            print(f"Description: {video['description']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 