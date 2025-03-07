from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DescriptionFinder:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant and BERT model"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
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
            
    def find_similar_descriptions(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Find video descriptions similar to the query text"""
        try:
            # Generate embedding for query text
            logger.info(f"Generating embedding for query: {query_text}")
            query_embedding = self._get_bert_embedding(query_text)
            
            # Search for similar descriptions
            logger.info("Searching for similar descriptions...")
            search_results = self.client.search(
                collection_name=self.descriptions_collection,
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            
            # Format results
            similar_descriptions = []
            for result in search_results:
                # Calculate cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                result_norm = np.linalg.norm(result.vector)
                cosine_similarity = np.dot(query_embedding, result.vector) / (query_norm * result_norm)
                
                similar_descriptions.append({
                    "video_id": result.payload["video_id"],
                    "description": result.payload["description"],
                    "model": result.payload.get("model_used", "N/A"),
                    "frames": result.payload.get("num_frames", 0),
                    "similarity_score": result.score,
                    "cosine_similarity": cosine_similarity
                })
                
            return similar_descriptions
            
        except Exception as e:
            logger.error(f"Error searching for similar descriptions: {str(e)}")
            raise
            
    def display_results(self, query_text: str, results: List[Dict]):
        """Display search results with detailed information"""
        print(f"\nSearch Query: {query_text}")
        print("\nSimilar Video Descriptions:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Video: {result['video_id']}")
            print(f"Description: {result['description']}")
            print(f"Model: {result['model']}")
            print(f"Frames: {result['frames']}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print(f"Cosine Similarity: {result['cosine_similarity']:.3f}")
            
        if not results:
            print("\nNo similar descriptions found!")

def search_descriptions(query_text: str, limit: int = 5, host: str = "localhost", port: int = 6333):
    """Helper function to search for similar descriptions"""
    try:
        finder = DescriptionFinder(host, port)
        results = finder.find_similar_descriptions(query_text, limit)
        finder.display_results(query_text, results)
        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def main():
    """Example usage of the description finder"""
    # Example query
    query = "phone"
    search_descriptions(query)

if __name__ == "__main__":
    main() 