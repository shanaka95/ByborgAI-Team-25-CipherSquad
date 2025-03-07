import os
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logger = logging.getLogger(__name__)

class QdrantManager:
    """
    A class to manage interactions with Qdrant vector database.
    Supports both local and cloud deployments of Qdrant.
    """
    
    def __init__(
        self,
        collection_name: str,
        vector_size: int = 768,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        location: Optional[str] = None,
        prefer_grpc: bool = True,
        timeout: int = 60
    ):
        """
        Initialize QdrantManager with connection parameters.
        
        Args:
            collection_name: Name of the collection to work with
            vector_size: Dimensionality of vectors (default 768 for many embedding models)
            url: URL for Qdrant cloud deployments
            api_key: API key for Qdrant cloud
            host: Host for local Qdrant instance
            port: Port for local Qdrant instance
            location: Path to local storage, if using persistent local storage
            prefer_grpc: Whether to prefer gRPC over HTTP
            timeout: Timeout for requests in seconds
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client based on provided parameters
        if url and api_key:
            # Cloud Qdrant
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )
            logger.info(f"Connected to Qdrant cloud at {url}")
        else:
            # Local Qdrant
            self.client = QdrantClient(
                host=host,
                port=port,
                location=location,
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )
            logger.info(f"Connected to local Qdrant at {host}:{port}")
    
    def create_collection(
        self,
        distance: str = "cosine",
        on_disk_payload: bool = True,
        hnsw_config: Optional[Dict[str, Any]] = None,
        optimizers_config: Optional[Dict[str, Any]] = None,
        force_recreate: bool = False
    ) -> bool:
        """
        Create a collection with specified parameters.
        
        Args:
            distance: Distance function ("cosine", "euclid", or "dot")
            on_disk_payload: Whether to store payload on disk instead of RAM
            hnsw_config: Configuration for HNSW index
            optimizers_config: Configuration for segment optimizers
            force_recreate: Whether to recreate collection if it exists
            
        Returns:
            bool: Success status
        """
        try:
            # Check if collection exists
            if self.collection_exists():
                if force_recreate:
                    logger.info(f"Collection {self.collection_name} exists, deleting due to force_recreate=True")
                    self.delete_collection()
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True
                
            # Convert distance string to Qdrant enum
            # Ensure lowercase for case-insensitive comparison
            distance = distance.lower()
            if distance == "cosine":
                distance_func = models.Distance.COSINE
            elif distance == "euclid" or distance == "euclidean":
                distance_func = models.Distance.EUCLID
            elif distance == "dot":
                distance_func = models.Distance.DOT
            else:
                logger.error(f"Unsupported distance function: {distance}")
                return False
            
            # Create vector configuration
            vector_config = models.VectorParams(
                size=self.vector_size,
                distance=distance_func
            )
            
            # Create collection with specified parameters
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config,
                on_disk_payload=on_disk_payload,
                hnsw_config=models.HnswConfigDiff(**hnsw_config) if hnsw_config else None,
                optimizers_config=models.OptimizersConfigDiff(**optimizers_config) if optimizers_config else None
            )
            
            logger.info(f"Created collection {self.collection_name} with vector size {self.vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {self.collection_name}: {str(e)}")
            return False
    
    def add_points(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Add points to the collection.
        
        Args:
            ids: List of point IDs (string or integers)
            vectors: List of vectors to add
            payloads: Optional list of payload dictionaries
            batch_size: Size of batches for insertion
            
        Returns:
            bool: Success status
        """
        try:
            if not payloads:
                payloads = [{} for _ in ids]
            
            # Ensure all inputs have the same length
            if not (len(ids) == len(vectors) == len(payloads)):
                raise ValueError("IDs, vectors, and payloads must have the same length")
            
            # Process in batches
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                batch_payloads = payloads[i:i+batch_size]
                
                points = [
                    models.PointStruct(
                        id=id_val,
                        vector=vector,
                        payload=payload
                    )
                    for id_val, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
                ]
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                
            logger.info(f"Added {len(ids)} points to collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding points to {self.collection_name}: {str(e)}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filter_conditions: Filter conditions for the search
            with_payload: Whether to return payloads
            score_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(**filter_conditions)
            
            search_params = models.SearchParams(
                hnsw_ef=128,  # Higher value gives more accurate but slower search
                exact=False  # Set to True for exact search (slower)
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=with_payload,
                search_params=search_params,
                score_threshold=score_threshold
            )
            
            # Convert Qdrant result objects to dictionaries
            processed_results = []
            for res in results:
                result_dict = {
                    "id": res.id,
                    "score": res.score,
                }
                if with_payload and res.payload:
                    result_dict["payload"] = res.payload
                processed_results.append(result_dict)
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching in {self.collection_name}: {str(e)}")
            return []
    
    def delete_points(self, point_ids: List[Union[str, int]]) -> bool:
        """
        Delete points from the collection.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids,
                ),
            )
            logger.info(f"Deleted {len(point_ids)} points from {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting points from {self.collection_name}: {str(e)}")
            return False
    
    def update_payload(
        self, 
        point_id: Union[str, int], 
        payload: Dict[str, Any], 
        replace: bool = False
    ) -> bool:
        """
        Update the payload of a point.
        
        Args:
            point_id: ID of the point to update
            payload: Payload data to update
            replace: If True, replace the entire payload; if False, update only specified fields
            
        Returns:
            bool: Success status
        """
        try:
            if replace:
                # Replace the entire payload
                self.client.overwrite_payload(
                    collection_name=self.collection_name,
                    points=[point_id],
                    payload=payload
                )
            else:
                # Update only specified fields
                self.client.set_payload(
                    collection_name=self.collection_name,
                    points=[point_id],
                    payload=payload
                )
            
            logger.info(f"Updated payload for point {point_id} in {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating payload in {self.collection_name}: {str(e)}")
            return False
    
    def get_points(
        self, 
        point_ids: List[Union[str, int]], 
        with_payload: bool = True, 
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve points by their IDs.
        
        Args:
            point_ids: List of point IDs to retrieve
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors
            
        Returns:
            List of retrieved points
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # Convert Qdrant result objects to dictionaries
            processed_results = []
            for res in results:
                result_dict = {
                    "id": res.id
                }
                if with_payload and hasattr(res, 'payload'):
                    result_dict["payload"] = res.payload
                if with_vectors and hasattr(res, 'vector'):
                    result_dict["vector"] = res.vector
                processed_results.append(result_dict)
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Error retrieving points from {self.collection_name}: {str(e)}")
            return []
    
    def count_points(self, filter_conditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Count points in the collection, optionally with a filter.
        
        Args:
            filter_conditions: Filter conditions
            
        Returns:
            Number of points
        """
        try:
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(**filter_conditions)
                
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=search_filter
            )
            
            return result.count
            
        except Exception as e:
            logger.error(f"Error counting points in {self.collection_name}: {str(e)}")
            return -1
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            
            # Convert to dict for easier access
            collection_info = {
                "name": info.name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": {
                        "size": info.config.params.size,
                        "distance": str(info.config.params.distance),
                    },
                    "hnsw_config": {
                        "m": info.config.hnsw_config.m,
                        "ef_construct": info.config.hnsw_config.ef_construct,
                    }
                }
            }
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Error getting info for {self.collection_name}: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {str(e)}")
            return False
    
    def scroll_points(
        self, 
        limit: int = 100, 
        with_payload: bool = True, 
        with_vectors: bool = False,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Scroll through points in the collection.
        
        Args:
            limit: Maximum number of points to return
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors
            filter_conditions: Filter conditions
            
        Returns:
            Tuple of (points list, next_offset or None if done)
        """
        try:
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(**filter_conditions)
                
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                filter=search_filter
            )
            
            # Convert Qdrant result objects to dictionaries
            processed_results = []
            for res in results:
                result_dict = {
                    "id": res.id
                }
                if with_payload and hasattr(res, 'payload'):
                    result_dict["payload"] = res.payload
                if with_vectors and hasattr(res, 'vector'):
                    result_dict["vector"] = res.vector
                processed_results.append(result_dict)
                
            return processed_results, next_offset
            
        except Exception as e:
            logger.error(f"Error scrolling points in {self.collection_name}: {str(e)}")
            return [], None
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            return any(collection.name == self.collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking if collection {self.collection_name} exists: {str(e)}")
            return False 