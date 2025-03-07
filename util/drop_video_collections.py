from qdrant_client import QdrantClient
import logging
import argparse
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CollectionManager:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
            
    def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection.vectors_count,
                "points_count": collection.points_count,
                "status": collection.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            raise
            
    def drop_collection(self, collection_name: str, force: bool = False):
        """Drop a collection after confirmation"""
        try:
            # Get collection info first
            info = self.get_collection_info(collection_name)
            
            if not force:
                # Show warning and get confirmation
                print(f"\nWARNING: You are about to drop collection '{collection_name}'")
                print(f"This collection contains {info['points_count']} points")
                print("This action cannot be undone!")
                
                confirmation = input("\nType 'YES' to confirm: ")
                if confirmation != "YES":
                    print("Operation cancelled.")
                    return
            
            # Drop the collection
            logger.info(f"Dropping collection: {collection_name}")
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' has been dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping collection {collection_name}: {str(e)}")
            raise

def main():
    """Drop video-related collections"""
    parser = argparse.ArgumentParser(description="Drop video-related collections from Qdrant")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    
    try:
        # Initialize manager
        manager = CollectionManager(args.host, args.port)
        
        # List current collections
        collections = manager.list_collections()
        print("\nCurrent collections:")
        for collection in collections:
            info = manager.get_collection_info(collection)
            print(f"- {collection}: {info['points_count']} points")
            
        # Define collections to drop
        video_collections = ["videos", "video_descriptions"]
        
        # Drop each collection
        for collection in video_collections:
            if collection in collections:
                manager.drop_collection(collection, args.force)
            else:
                logger.info(f"Collection '{collection}' does not exist")
        
        # Show remaining collections
        remaining = manager.list_collections()
        if remaining:
            print("\nRemaining collections:")
            for collection in remaining:
                info = manager.get_collection_info(collection)
                print(f"- {collection}: {info['points_count']} points")
        else:
            print("\nNo collections remaining")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 