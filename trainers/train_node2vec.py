import os
import logging
import csv
from typing import List, Dict, Tuple, Set
import pickle
import random
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = {
    'networkx': False,
    'gensim': False,
    'numpy': False,
    'pandas': False,
    'matplotlib': False,
    'sklearn': False,
    'tqdm': False
}

# Try to import each package
try:
    import networkx as nx
    required_packages['networkx'] = True
except ImportError:
    logger.warning("networkx package not found")

try:
    from gensim.models import Word2Vec
    required_packages['gensim'] = True
except ImportError:
    logger.warning("gensim package not found")

try:
    import numpy as np
    required_packages['numpy'] = True
except ImportError:
    logger.warning("numpy package not found")

try:
    import pandas as pd
    required_packages['pandas'] = True
except ImportError:
    logger.warning("pandas package not found")

try:
    import matplotlib.pyplot as plt
    required_packages['matplotlib'] = True
except ImportError:
    logger.warning("matplotlib package not found")

try:
    from sklearn.manifold import TSNE
    required_packages['sklearn'] = True
except ImportError:
    logger.warning("sklearn package not found")

try:
    from tqdm import tqdm
    required_packages['tqdm'] = True
except ImportError:
    # Define a simple tqdm replacement if not available
    def tqdm(iterable, **kwargs):
        return iterable
    logger.warning("tqdm package not found, using simple replacement")

# Check if all required packages are available
all_packages_available = all(required_packages.values())
if not all_packages_available:
    missing_packages = [pkg for pkg, available in required_packages.items() if not available]
    logger.warning(f"Missing packages: {', '.join(missing_packages)}")
    logger.warning("To install missing packages, try one of the following:")
    logger.warning("1. Create a virtual environment: python3 -m venv myenv && source myenv/bin/activate")
    logger.warning("2. Install packages: pip install networkx gensim matplotlib scikit-learn pandas numpy tqdm")
    logger.warning("3. Or use system packages: sudo apt-get install python3-networkx python3-gensim python3-matplotlib python3-sklearn python3-pandas python3-numpy python3-tqdm")

class Node2VecTrainer:
    """
    Class to train a Node2Vec model on user session data.
    Node2Vec learns embeddings for videos based on how they co-occur in user sessions.
    """
    
    def __init__(
        self, 
        sessions_file: str = "user-sessions/user_sessions.csv",
        output_dir: str = "models",
        dimensions: int = 128,
        walk_length: int = 10,
        num_walks: int = 100,
        p: float = 1.0,
        q: float = 1.0,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 5
    ):
        """
        Initialize the Node2Vec trainer.
        
        Args:
            sessions_file: Path to the user sessions CSV file
            output_dir: Directory to save the model and results
            dimensions: Dimensionality of the embeddings
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (1 = neutral, <1 = depth-first, >1 = breadth-first)
            q: In-out parameter (1 = neutral, <1 = return to source, >1 = explore outward)
            window: Context window size for Word2Vec
            min_count: Minimum count of node occurrences
            workers: Number of parallel workers
            epochs: Number of training epochs
        """
        self.sessions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), sessions_file)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Node2Vec parameters
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        
        # Initialize graph and model
        if 'networkx' in globals():
            self.graph = nx.Graph()
        else:
            self.graph = None
        self.model = None
        self.video_embeddings = {}
        
    def load_sessions(self):
        """Load user sessions from CSV file"""
        logger.info(f"Loading user sessions from {self.sessions_file}")
        try:
            if 'pandas' in globals():
                df = pd.read_csv(self.sessions_file)
                logger.info(f"Loaded {len(df)} user sessions")
                return df
            else:
                # Fallback to using csv module
                sessions = []
                with open(self.sessions_file, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Skip header
                    for row in reader:
                        sessions.append(row)
                logger.info(f"Loaded {len(sessions)} user sessions")
                return sessions
        except Exception as e:
            logger.error(f"Error loading sessions file: {str(e)}")
            raise
    
    def build_graph(self, sessions_data):
        """
        Build a graph from user sessions where:
        - Nodes are videos
        - Edges connect videos watched in the same session
        - Edge weights represent co-occurrence frequency
        """
        if not all_packages_available:
            logger.error("Cannot build graph: required packages are missing")
            return None
            
        logger.info("Building video co-occurrence graph")
        
        # Initialize graph
        G = nx.Graph()
        
        # Process each user session
        if isinstance(sessions_data, pd.DataFrame):
            # If sessions_data is a pandas DataFrame
            for _, row in tqdm(sessions_data.iterrows(), total=len(sessions_data), desc="Processing sessions"):
                # Get videos in this session (skip empty cells)
                videos = [v for v in row[1:] if isinstance(v, str) and v]
                
                self._add_session_to_graph(G, videos)
        else:
            # If sessions_data is a list from csv module
            for row in tqdm(sessions_data, desc="Processing sessions"):
                # Get videos in this session (skip empty cells)
                videos = [v for v in row[1:] if v]
                
                self._add_session_to_graph(G, videos)
        
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _add_session_to_graph(self, G, videos):
        """Add a session's videos to the graph"""
        # Add nodes and edges for videos in this session
        for i in range(len(videos)):
            if not G.has_node(videos[i]):
                G.add_node(videos[i])
            
            # Connect each video to all other videos in the session
            for j in range(i+1, len(videos)):
                if videos[i] != videos[j]:  # Avoid self-loops
                    # If edge exists, increment weight, otherwise create with weight 1
                    if G.has_edge(videos[i], videos[j]):
                        G[videos[i]][videos[j]]['weight'] += 1
                    else:
                        G.add_edge(videos[i], videos[j], weight=1)
    
    def generate_walks(self, G):
        """
        Generate random walks for Node2Vec training.
        This is a simplified implementation of Node2Vec random walks.
        """
        if not all_packages_available:
            logger.error("Cannot generate walks: required packages are missing")
            return []
            
        logger.info(f"Generating {self.num_walks} walks of length {self.walk_length} for each node")
        walks = []
        
        nodes = list(G.nodes())
        
        for _ in tqdm(range(self.num_walks), desc="Generating walks"):
            # Shuffle nodes for each round of walks
            random.shuffle(nodes)
            
            for node in nodes:
                # Start a walk from this node
                walk = [node]
                
                for _ in range(self.walk_length - 1):
                    current = walk[-1]
                    neighbors = list(G.neighbors(current))
                    
                    if not neighbors:
                        break
                    
                    # Get weights for neighbors
                    weights = [G[current][neighbor].get('weight', 1) for neighbor in neighbors]
                    
                    # Choose next node based on edge weights
                    next_node = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(next_node)
                
                if len(walk) > 1:  # Only add walks with at least 2 nodes
                    walks.append(walk)
        
        logger.info(f"Generated {len(walks)} walks")
        return walks
    
    def train_model(self, walks):
        """Train Word2Vec model on the generated walks"""
        if not all_packages_available:
            logger.error("Cannot train model: required packages are missing")
            return None
            
        logger.info(f"Training Word2Vec model with {self.dimensions} dimensions")
        
        model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=self.epochs
        )
        
        logger.info("Model training completed")
        return model
    
    def save_model(self, model, filename: str = "node2vec_model"):
        """Save the trained model and embeddings"""
        if model is None:
            logger.error("Cannot save model: model is None")
            return None
            
        # Save Word2Vec model
        model_path = os.path.join(self.output_dir, f"{filename}.model")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Extract and save video embeddings as a dictionary
        video_embeddings = {}
        for video in model.wv.index_to_key:
            video_embeddings[video] = model.wv[video]
        
        embeddings_path = os.path.join(self.output_dir, f"{filename}_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(video_embeddings, f)
        logger.info(f"Embeddings saved to {embeddings_path}")
        
        return model_path
    
    def visualize_embeddings(self, model, filename: str = "video_embeddings_tsne.png"):
        """Visualize video embeddings using t-SNE"""
        if not all_packages_available or model is None:
            logger.error("Cannot visualize embeddings: required packages are missing or model is None")
            return
            
        logger.info("Visualizing embeddings with t-SNE")
        
        # Get embeddings and labels
        embeddings = []
        labels = []
        
        for video in model.wv.index_to_key:
            embeddings.append(model.wv[video])
            labels.append(video)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # Add labels for a subset of points to avoid overcrowding
        max_labels = min(50, len(labels))
        indices = np.random.choice(len(labels), max_labels, replace=False)
        
        for i in indices:
            plt.annotate(
                labels[i].replace('.avi', ''),
                xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 2),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.title(f"t-SNE Visualization of Video Embeddings")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Visualization saved to {output_path}")
        
        plt.close()
    
    def find_similar_videos(self, video_id: str, top_n: int = 5):
        """Find similar videos using the trained model"""
        if not all_packages_available or self.model is None:
            logger.error("Cannot find similar videos: required packages are missing or model is None")
            return []
            
        if video_id not in self.model.wv:
            logger.error(f"Video {video_id} not found in model vocabulary")
            return []
        
        similar_videos = self.model.wv.most_similar(video_id, topn=top_n)
        return similar_videos
    
    def train(self):
        """Complete training pipeline"""
        if not all_packages_available:
            logger.error("Cannot train model: required packages are missing")
            logger.info("Please install the required packages and try again")
            return None
            
        # Load sessions data
        sessions_data = self.load_sessions()
        
        # Build graph
        self.graph = self.build_graph(sessions_data)
        if self.graph is None:
            return None
        
        # Generate random walks
        walks = self.generate_walks(self.graph)
        if not walks:
            return None
        
        # Train model
        self.model = self.train_model(walks)
        if self.model is None:
            return None
        
        # Save model and embeddings
        self.save_model(self.model)
        
        # Visualize embeddings
        self.visualize_embeddings(self.model)
        
        return self.model

def main():
    """Main function to train the Node2Vec model"""
    if not all_packages_available:
        logger.error("Cannot run Node2Vec training: required packages are missing")
        return
        
    # Initialize trainer with default parameters
    trainer = Node2VecTrainer(
        sessions_file="user-sessions/user_sessions.csv",
        output_dir="models",
        dimensions=128,
        walk_length=10,
        num_walks=100,
        p=1.0,
        q=1.0,
        window=5,
        min_count=1,
        workers=4,
        epochs=5
    )
    
    # Train model
    model = trainer.train()
    
    if model is not None:
        # Save model to file with a timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f"node2vec_model_{timestamp}"
        
        # Save the model using the trainer's save_model method
        model_path = trainer.save_model(model, model_filename)
        logger.info(f"Model successfully saved to {model_path}")
        
        # Save embeddings dictionary separately for easy access
        embeddings_dict = {}
        for video in model.wv.index_to_key:
            embeddings_dict[video] = model.wv[video].tolist()  # Convert numpy array to list for better serialization
        
        embeddings_json_path = os.path.join(trainer.output_dir, f"{model_filename}_embeddings.json")
        try:
            with open(embeddings_json_path, 'w') as f:
                json.dump(embeddings_dict, f)
            logger.info(f"Embeddings saved as JSON to {embeddings_json_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings as JSON: {str(e)}")
        
        # Example: Find similar videos for a specific video
        example_video = "sceneclipautoautotrain00301.avi"  # From the user session we saw
        similar_videos = trainer.find_similar_videos(example_video, top_n=5)
        
        print(f"\nVideos similar to {example_video}:")
        for video, similarity in similar_videos:
            print(f"  - {video}: {similarity:.4f}")
        
        print(f"\nModel and embeddings saved to {trainer.output_dir}")
        print(f"Model file: {model_filename}.model")
        print(f"Embeddings file: {model_filename}_embeddings.pkl")
        if os.path.exists(embeddings_json_path):
            print(f"JSON embeddings file: {model_filename}_embeddings.json")

if __name__ == "__main__":
    main() 