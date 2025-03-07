import os
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import pickle
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. LambdaMART reranking will be disabled.")
    LIGHTGBM_AVAILABLE = False

class VideoRecommender:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize the video recommender with BERT model and Qdrant client"""
        # Connect to Qdrant
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.descriptions_collection = "video_descriptions"
        
        # Initialize BERT model and tokenizer
        logger.info("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"BERT model loaded successfully (using {self.device})")
        
        # Load Node2Vec model embeddings if available
        self.node2vec_embeddings = None
        self.load_node2vec_embeddings()
        
        # Load LambdaMART model if available
        self.lambdamart_model = None
        self.feature_names = None
        if LIGHTGBM_AVAILABLE:
            self.load_lambdamart_model()
    
    def load_node2vec_embeddings(self):
        """Load Node2Vec embeddings from the most recent model file"""
        try:
            # Look for Node2Vec embedding files
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            # First try JSON files (easier to load)
            json_files = glob.glob(os.path.join(models_dir, "*_embeddings.json"))
            if json_files:
                # Get the most recent file
                latest_json = max(json_files, key=os.path.getctime)
                logger.info(f"Loading Node2Vec embeddings from {latest_json}")
                
                with open(latest_json, 'r') as f:
                    self.node2vec_embeddings = json.load(f)
                logger.info(f"Loaded Node2Vec embeddings for {len(self.node2vec_embeddings)} videos")
                return
            
            # If no JSON files, try pickle files
            pkl_files = glob.glob(os.path.join(models_dir, "*_embeddings.pkl"))
            if pkl_files:
                # Get the most recent file
                latest_pkl = max(pkl_files, key=os.path.getctime)
                logger.info(f"Loading Node2Vec embeddings from {latest_pkl}")
                
                with open(latest_pkl, 'rb') as f:
                    self.node2vec_embeddings = pickle.load(f)
                logger.info(f"Loaded Node2Vec embeddings for {len(self.node2vec_embeddings)} videos")
                return
                
            logger.warning("No Node2Vec embedding files found")
            
        except Exception as e:
            logger.error(f"Error loading Node2Vec embeddings: {str(e)}")
            self.node2vec_embeddings = None
    
    def load_lambdamart_model(self):
        """Load LambdaMART model for reranking"""
        try:
            # Look for LambdaMART model files
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            model_path = os.path.join(models_dir, "lambdamart_model.txt")
            feature_path = os.path.join(models_dir, "feature_names.pkl")
            
            if os.path.exists(model_path) and os.path.exists(feature_path):
                # Load feature names
                with open(feature_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                
                # Load model
                self.lambdamart_model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded LambdaMART model from {model_path}")
                logger.info(f"Model features: {self.feature_names}")
                return
                
            logger.warning("LambdaMART model files not found")
            
        except Exception as e:
            logger.error(f"Error loading LambdaMART model: {str(e)}")
            self.lambdamart_model = None
    
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
    
    def get_video_description(self, video_name: str) -> Dict[str, Any]:
        """Retrieve the description for a video from Qdrant"""
        try:
            # Find the video's description
            video_results = self.client.scroll(
                collection_name=self.descriptions_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_name)
                        )
                    ]
                ),
                limit=1
            )
            
            points = video_results[0]
            if not points:
                logger.warning(f"No description found for video: {video_name}")
                return {"error": f"No description found for {video_name}"}
            
            return points[0].payload
            
        except Exception as e:
            logger.error(f"Error retrieving description for {video_name}: {str(e)}")
            return {"error": str(e)}
    
    def find_similar_videos(self, description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find videos with similar descriptions using BERT embeddings"""
        try:
            # Generate embedding for the description
            query_embedding = self._get_bert_embedding(description)
            
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
                    "video_id": result.payload.get("video_id", "unknown"),
                    "description": result.payload.get("description", "No description available"),
                    "similarity_score": result.score,
                    "source": "bert"
                })
                
            return similar_videos
            
        except Exception as e:
            logger.error(f"Error searching for similar videos: {str(e)}")
            return []
    
    def find_similar_videos_node2vec(self, video_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar videos using Node2Vec embeddings"""
        if not self.node2vec_embeddings or video_id not in self.node2vec_embeddings:
            logger.warning(f"No Node2Vec embeddings available for {video_id}")
            return []
        
        try:
            # Get embedding for the video
            video_embedding = self.node2vec_embeddings[video_id]
            
            # Calculate cosine similarity with all other videos
            similarities = []
            for other_id, other_embedding in self.node2vec_embeddings.items():
                if other_id != video_id:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(video_embedding, other_embedding)
                    similarities.append((other_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N results
            top_results = similarities[:limit]
            
            # Format results
            similar_videos = []
            for other_id, similarity in top_results:
                # Try to get description if available
                description = "No description available"
                try:
                    video_info = self.get_video_description(other_id)
                    if "error" not in video_info:
                        description = video_info.get("description", description)
                except:
                    pass
                
                similar_videos.append({
                    "video_id": other_id,
                    "description": description,
                    "similarity_score": similarity,
                    "source": "node2vec"
                })
            
            return similar_videos
            
        except Exception as e:
            logger.error(f"Error finding similar videos with Node2Vec: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
            
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
            
        return dot_product / (norm_vec1 * norm_vec2)
    
    def rerank_with_lambdamart(self, video_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank video pairs using LambdaMART model"""
        if not LIGHTGBM_AVAILABLE or self.lambdamart_model is None or not video_pairs:
            logger.warning("LambdaMART reranking not available or no videos to rerank")
            return video_pairs
        
        try:
            # Prepare features for LambdaMART
            features = []
            
            for video in video_pairs:
                # Extract features needed by the model
                video_features = []
                
                # Add description similarity
                video_features.append(video["similarity_score"])
                
                # Add category match (if available)
                if "category_match" in video:
                    category_match_numeric = 1 if video["category_match"] == "Y" else 0
                    video_features.append(category_match_numeric)
                else:
                    # If category match is not available, use a default value
                    video_features.append(0)
                
                features.append(video_features)
            
            # Convert to numpy array
            features_array = np.array(features)
            
            # Make predictions
            scores = self.lambdamart_model.predict(features_array)
            
            # Add scores to videos
            for i, video in enumerate(video_pairs):
                video["lambdamart_score"] = float(scores[i])
            
            # Sort by LambdaMART score (descending)
            reranked_videos = sorted(video_pairs, key=lambda x: x["lambdamart_score"], reverse=True)
            
            return reranked_videos
            
        except Exception as e:
            logger.error(f"Error reranking with LambdaMART: {str(e)}")
            return video_pairs
    
    def get_recommendations_for_session(self, session_videos: List[str], recommendations_per_video: int = 5) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Get recommendations based on a list of previously watched videos"""
        recommendations = {}
        
        for video_name in session_videos:
            video_recommendations = {
                "bert": [],
                "node2vec": [],
                "combined": []
            }
            
            # Get video description
            video_info = self.get_video_description(video_name)
            
            # Get BERT-based recommendations
            if "error" not in video_info:
                # Get description text
                description = video_info.get("description", "")
                if description:
                    # Find similar videos using BERT
                    similar_videos = self.find_similar_videos(description, limit=recommendations_per_video * 2)
                    
                    # Filter out the original video from recommendations
                    video_recommendations["bert"] = [
                        video for video in similar_videos 
                        if video["video_id"] != video_name
                    ][:recommendations_per_video]
            
            # Get Node2Vec-based recommendations
            if self.node2vec_embeddings:
                node2vec_recommendations = self.find_similar_videos_node2vec(video_name, limit=recommendations_per_video * 2)
                video_recommendations["node2vec"] = node2vec_recommendations[:recommendations_per_video]
            
            # Combine all recommendations for reranking
            all_recommendations = []
            all_recommendations.extend(video_recommendations["bert"])
            all_recommendations.extend(video_recommendations["node2vec"])
            
            # Remove duplicates (prefer BERT if duplicate)
            seen_videos = set()
            unique_recommendations = []
            
            for rec in all_recommendations:
                video_id = rec["video_id"]
                if video_id not in seen_videos and video_id != video_name:
                    seen_videos.add(video_id)
                    unique_recommendations.append(rec)
            
            # Rerank with LambdaMART if available
            if LIGHTGBM_AVAILABLE and self.lambdamart_model is not None:
                reranked_recommendations = self.rerank_with_lambdamart(unique_recommendations)
                # Take top 10 after reranking
                video_recommendations["combined"] = reranked_recommendations[:10]
            else:
                # If LambdaMART is not available, just combine and take top 10
                video_recommendations["combined"] = unique_recommendations[:10]
            
            recommendations[video_name] = video_recommendations
            
        return recommendations

def main():
    # Set user ID
    user_id = "d5d292d0-ba64-4771-86fe-32a6a44716f3"
    
    # Set previously visited sessions
    previous_sessions = [
        "sceneclipautoautotrain00301.avi",
        "sceneclipautoautotrain00213.avi",
        "scenecliptest00006.avi"
    ]
    
    # Initialize recommender
    recommender = VideoRecommender()
    
    # Get recommendations for each previously visited session
    recommendations = recommender.get_recommendations_for_session(previous_sessions)
    
    # Print recommendations
    print(f"Recommendations for user {user_id}:")
    print("=" * 50)
    
    for video_name, rec_sources in recommendations.items():
        print(f"\nBased on your interest in: {video_name}")
        
        # Print BERT-based recommendations
        bert_recs = rec_sources.get("bert", [])
        if bert_recs:
            print("\n  BERT-based recommendations (content similarity):")
            for i, video in enumerate(bert_recs, 1):
                print(f"  {i}. {video['video_id']} (Similarity: {video['similarity_score']:.2f})")
                print(f"     Description: {video['description'][:100]}...")
        else:
            print("\n  No BERT-based recommendations found.")
        
        # Print Node2Vec-based recommendations
        node2vec_recs = rec_sources.get("node2vec", [])
        if node2vec_recs:
            print("\n  Node2Vec-based recommendations (user behavior similarity):")
            for i, video in enumerate(node2vec_recs, 1):
                print(f"  {i}. {video['video_id']}")
                if video['description'] != "No description available":
                    print(f"     Description: {video['description'][:100]}...")
        else:
            print("\n  No Node2Vec-based recommendations found.")
        
        # Print LambdaMART reranked recommendations
        combined_recs = rec_sources.get("combined", [])
        if combined_recs:
            print("\n  TOP 10 RECOMMENDATIONS (LambdaMART reranked):")
            for i, video in enumerate(combined_recs, 1):
                score_info = f"(LambdaMART - "
                score_info += f"{video['source'].upper()})"
                print(f"  {i}. {video['video_id']} {score_info}")
                if video['description'] != "No description available":
                    print(f"     Description: {video['description'][:100]}...")
        else:
            print("\n  No combined recommendations found.")

if __name__ == "__main__":
    main()
