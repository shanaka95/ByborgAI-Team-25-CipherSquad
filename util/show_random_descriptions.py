from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import random
from datetime import datetime
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DescriptionViewer:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize connection to Qdrant"""
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.descriptions_collection = "video_descriptions"
        
    def get_random_descriptions(self, num_records: int = 10) -> List[Dict]:
        """Fetch random description records with embeddings from the collection"""
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
                with_vectors=True  # Also fetch embeddings
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
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding": record.vector  # Store the embedding
                })
            
            return formatted_records
            
        except Exception as e:
            logger.error(f"Error fetching random descriptions: {str(e)}")
            raise
            
    def verify_embeddings(self, descriptions: List[Dict]) -> Dict:
        """Verify embeddings quality and statistics"""
        try:
            if not descriptions:
                return {}
                
            # Convert embeddings to numpy array
            embeddings = np.array([desc["embedding"] for desc in descriptions])
            
            # Calculate basic statistics
            stats = {
                "embedding_dim": embeddings.shape[1],
                "num_samples": len(embeddings),
                "mean_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
                "std_norm": np.std(np.linalg.norm(embeddings, axis=1)),
                "mean_values": np.mean(embeddings),
                "std_values": np.std(embeddings),
                "zero_vectors": np.sum(np.all(embeddings == 0, axis=1)),
                "nan_vectors": np.sum(np.any(np.isnan(embeddings), axis=1))
            }
            
            # Calculate cosine similarities between all pairs
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            similarities = np.dot(normalized, normalized.T)
            
            # Add similarity statistics
            stats.update({
                "mean_similarity": np.mean(similarities[np.triu_indices_from(similarities, k=1)]),
                "min_similarity": np.min(similarities[np.triu_indices_from(similarities, k=1)]),
                "max_similarity": np.max(similarities[np.triu_indices_from(similarities, k=1)])
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error verifying embeddings: {str(e)}")
            raise
            
    def visualize_embeddings(self, descriptions: List[Dict], output_file: str = "embedding_visualization.png"):
        """Create visualization of embeddings using PCA and t-SNE"""
        try:
            if not descriptions:
                logger.warning("No descriptions to visualize")
                return
                
            embeddings = np.array([desc["embedding"] for desc in descriptions])
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # PCA visualization
            logger.info("Computing PCA...")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax1)
            ax1.set_title("PCA Visualization")
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            
            # t-SNE visualization
            logger.info("Computing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(embeddings)
            
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], ax=ax2)
            ax2.set_title("t-SNE Visualization")
            
            # Add video IDs as tooltips
            for i, desc in enumerate(descriptions):
                ax1.annotate(desc["video_id"], 
                           (pca_result[i, 0], pca_result[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
                ax2.annotate(desc["video_id"], 
                           (tsne_result[i, 0], tsne_result[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {str(e)}")
            raise
            
    def display_descriptions(self, descriptions: List[Dict]):
        """Display the descriptions and embedding statistics"""
        if not descriptions:
            print("\nNo descriptions found!")
            return
            
        # Verify embeddings
        stats = self.verify_embeddings(descriptions)
        
        # Print embedding statistics
        print("\nEmbedding Statistics:")
        print(f"Dimension: {stats['embedding_dim']}")
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Mean vector norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}")
        print(f"Mean value: {stats['mean_values']:.3f} ± {stats['std_values']:.3f}")
        print(f"Zero vectors: {stats['zero_vectors']}")
        print(f"NaN vectors: {stats['nan_vectors']}")
        print(f"\nSimilarity Statistics:")
        print(f"Mean similarity: {stats['mean_similarity']:.3f}")
        print(f"Min similarity: {stats['min_similarity']:.3f}")
        print(f"Max similarity: {stats['max_similarity']:.3f}")
        
        # Print descriptions
        print("\nRandom Video Descriptions:")
        for i, record in enumerate(descriptions, 1):
            print(f"\n{i}. Video: {record['video_id']}")
            print(f"Description: {record['description']}")
            print(f"Model: {record['model']}")
            print(f"Frames: {record['frames']}")
            print(f"Timestamp: {record['timestamp']}")
            print(f"Embedding norm: {np.linalg.norm(record['embedding']):.3f}")
        
        # Create visualizations
        self.visualize_embeddings(descriptions)

def main():
    """Display random video descriptions and analyze embeddings"""
    try:
        # Initialize viewer
        viewer = DescriptionViewer()
        
        # Get and display random descriptions with embeddings
        descriptions = viewer.get_random_descriptions(num_records=20)  # Increased for better visualization
        viewer.display_descriptions(descriptions)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 