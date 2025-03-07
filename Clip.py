import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from typing import List, Union, Optional, Tuple, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)

class ClipEncoder:
    """
    A class to handle image and text encoding using CLIP model via SentenceTransformer.
    """
    
    AVAILABLE_MODELS = [
        "clip-ViT-B-32", 
        "clip-ViT-B-16", 
        "clip-ViT-L-14",
    ]
    
    def __init__(
        self, 
        model_name: str = "clip-ViT-B-16",
        device: Optional[str] = None,
    ):
        """
        Initialize the CLIP encoder with specified model.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to use (cuda or cpu). If None, will use cuda if available.
        """
        # Add 'clip-' prefix if not already present
        if model_name not in self.AVAILABLE_MODELS and not model_name.startswith('clip-'):
            model_name = f"clip-{model_name}"
            
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model {model_name} not in known models: {self.AVAILABLE_MODELS}. Attempting anyway.")
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading CLIP model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Get vector dimensions from model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"CLIP model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_images(
        self, 
        images: Union[List[Image.Image], List[str], Image.Image, str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to feature vectors using CLIP.
        
        Args:
            images: List of PIL images, list of file paths, single image, or single file path
            batch_size: Batch size for processing multiple images
            normalize: Whether to normalize the embeddings (SentenceTransformer does this by default)
            
        Returns:
            Array of image embeddings
        """
        # Convert single image to list
        if isinstance(images, (str, Image.Image)):
            images = [images]
            
        # Load images if paths are provided
        processed_images = []
        for img in images:
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert("RGB")
                except Exception as e:
                    logger.error(f"Error loading image {img}: {str(e)}")
                    continue
            
            processed_images.append(img)
        
        if not processed_images:
            return np.array([])
            
        # Use SentenceTransformer's encode method with show_progress_bar for larger batches
        show_progress = len(processed_images) > 10
        embeddings = self.model.encode(
            processed_images, 
            batch_size=batch_size, 
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def encode_texts(
        self, 
        texts: Union[List[str], str],
        batch_size: int = 256,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to feature vectors using CLIP.
        
        Args:
            texts: List of texts or single text string
            batch_size: Batch size for processing multiple texts
            normalize: Whether to normalize the embeddings (SentenceTransformer does this by default)
            
        Returns:
            Array of text embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
            
        # Use SentenceTransformer's encode method
        show_progress = len(texts) > 50
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between image and text features.
        
        Args:
            image_features: Image embeddings
            text_features: Text embeddings
            
        Returns:
            Similarity scores
        """
        # Use SentenceTransformer's util.cos_sim for cosine similarity
        similarities = util.cos_sim(
            torch.from_numpy(image_features), 
            torch.from_numpy(text_features)
        )
        
        # Convert to numpy and apply temperature scaling similar to CLIP (100.0 * ... softmax)
        similarities = (100.0 * similarities).softmax(dim=-1).cpu().numpy()
        return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded CLIP model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model.get_config_dict().get('model_name', 'unknown'),
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "state": "loaded"
        }
