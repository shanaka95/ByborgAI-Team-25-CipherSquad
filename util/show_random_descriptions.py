from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import random
from datetime import datetime
from typing import List, Dict
from tabulate import tabulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DescriptionViewer:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.descriptions_collection = "video_descriptions"
        
    def get_random_descriptions(self, num_records: int = 10) -> List[Dict]:
        """Fetch random description records from the collection"""
        try:
            # First, get total count of records
            collection_info = self.client.get_collection(self.descriptions_collection)
            total_records = collection_info.points_count
            
            if total_records == 0:
                logger.warning("No descriptions found in the collection")
                return []
            
            # Get all records (we'll randomly sample from these)
            all_records = self.client.scroll(
                collection_name=self.descriptions_collection,
                limit=total_records,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Randomly sample records
            sample_size = min(num_records, len(all_records))
            sampled_records = random.sample(all_records, sample_size)
            
            # Format the records
            formatted_records = []
            for record in sampled_records:
                payload = record.payload
                formatted_records.append({
                    "video_id": payload.get("video_id", "N/A"),
                    "description": payload.get("description", "N/A"),
                    "model": payload.get("model_used", "N/A"),
                    "frames": payload.get("num_frames", 0),
                    "timestamp": datetime.fromtimestamp(
                        payload.get("timestamp", 0)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return formatted_records
            
        except Exception as e:
            logger.error(f"Error fetching random descriptions: {str(e)}")
            raise
            
    def display_descriptions(self, descriptions: List[Dict]):
        """Display the descriptions in a formatted table"""
        if not descriptions:
            print("\nNo descriptions found!")
            return
            
        # Prepare table data
        headers = ["Video", "Description", "Model", "Frames", "Timestamp"]
        table_data = [
            [
                record["video_id"],
                record["description"][:100] + "..." if len(record["description"]) > 100 else record["description"],
                record["model"],
                record["frames"],
                record["timestamp"]
            ]
            for record in descriptions
        ]
        
        # Print table
        print("\nRandom Video Descriptions:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print full descriptions
        print("\nFull Descriptions:")
        for i, record in enumerate(descriptions, 1):
            print(f"\n{i}. Video: {record['video_id']}")
            print(f"Description: {record['description']}")

def main():
    """Display random video descriptions"""
    try:
        # Initialize viewer
        viewer = DescriptionViewer()
        
        # Get and display random descriptions
        descriptions = viewer.get_random_descriptions()
        viewer.display_descriptions(descriptions)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 