import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, Set, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class UserSessionVisualizer:
    def __init__(self, data_file: str = "../data/user_sessions.csv"):
        """Initialize the visualizer with the data file path"""
        self.data_file = data_file
        self.df = self._load_data()
        self.graph = nx.Graph()
        
    def _load_data(self) -> pd.DataFrame:
        """Load user session data from CSV"""
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
                
            logger.info(f"Loading data from {self.data_file}")
            df = pd.read_csv(self.data_file)
            
            # Verify expected columns exist
            expected_cols = ['user_id'] + [f'video{i}' for i in range(1, 6)]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(df)} user sessions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _build_video_connections(self) -> Dict[Tuple[str, str], int]:
        """Build connections between videos based on user sessions"""
        connections = defaultdict(int)
        
        # Get video columns
        video_cols = [col for col in self.df.columns if col.startswith('video')]
        
        # Count co-occurrences of videos
        logger.info("Analyzing video connections...")
        total_connections = 0
        
        for _, row in self.df.iterrows():
            videos = row[video_cols].dropna().tolist()
            for i in range(len(videos)):
                for j in range(i + 1, len(videos)):
                    # Sort video names to ensure consistent edge keys
                    v1, v2 = sorted([videos[i], videos[j]])
                    connections[(v1, v2)] += 1
                    total_connections += 1
                    
        logger.info(f"Found {total_connections} total connections between videos")
        return connections
        
    def create_graph(self, min_weight: int = 10):
        """Create graph from video connections"""
        try:
            # Get video connections
            connections = self._build_video_connections()
            
            if not connections:
                raise ValueError("No video connections found in the data")
            
            # Add edges to graph
            logger.info("Creating graph...")
            edges_added = 0
            for (v1, v2), weight in connections.items():
                if weight >= min_weight:  # Only add edges with sufficient weight
                    self.graph.add_edge(v1, v2, weight=weight)
                    edges_added += 1
                    
            if edges_added == 0:
                raise ValueError(f"No connections met the minimum weight threshold of {min_weight}. Try lowering the threshold.")
                    
            logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise
            
    def visualize(self, output_file: str = "user_session_graph.png"):
        """Visualize the graph"""
        try:
            if self.graph.number_of_nodes() == 0:
                raise ValueError("Graph is empty. Check if create_graph() was successful.")
                
            logger.info("Generating visualization...")
            
            # Create figure
            plt.figure(figsize=(20, 20))
            
            # Calculate node sizes based on degree centrality
            centrality = nx.degree_centrality(self.graph)
            node_sizes = [20000 * centrality[node] for node in self.graph.nodes()]
            
            # Calculate edge widths based on normalized weights
            max_weight = max(data['weight'] for _, _, data in self.graph.edges(data=True))
            edge_widths = [2.0 * self.graph[u][v]['weight'] / max_weight for u, v in self.graph.edges()]
            
            # Calculate layout
            logger.info("Computing layout...")
            pos = nx.spring_layout(self.graph, k=2, iterations=100)
            
            # Draw the graph
            logger.info("Drawing graph elements...")
            nx.draw_networkx_nodes(self.graph, pos, 
                                 node_size=node_sizes,
                                 node_color='lightblue',
                                 alpha=0.7)
            
            nx.draw_networkx_edges(self.graph, pos,
                                 width=edge_widths,
                                 alpha=0.5,
                                 edge_color='gray')
            
            # Add labels with smaller font for less connected nodes
            labels = {node: node.split('.')[0] for node in self.graph.nodes()}  # Remove file extension
            font_sizes = {node: min(12, 8 + 10 * centrality[node]) for node in self.graph.nodes()}
            
            for node, label in labels.items():
                plt.annotate(label,
                           xy=pos[node],
                           xytext=(0, 0),
                           textcoords="offset points",
                           fontsize=font_sizes[node],
                           ha='center',
                           va='center')
            
            plt.title("Video Co-occurrence Network\nNode size: Connection centrality, Edge width: Co-occurrence frequency",
                     pad=20, fontsize=14)
            plt.axis('off')
            
            # Save the visualization
            output_path = os.path.join("../data", output_file)
            logger.info(f"Saving visualization to {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved successfully")
            
            # Generate statistics
            self._print_statistics()
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            raise

    def _print_statistics(self):
        """Print graph statistics"""
        print("\nGraph Statistics:")
        print(f"Number of videos (nodes): {self.graph.number_of_nodes()}")
        print(f"Number of connections (edges): {self.graph.number_of_edges()}")
        
        # Find most central videos
        centrality = nx.degree_centrality(self.graph)
        top_videos = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nMost Central Videos:")
        for video, cent in top_videos:
            print(f"{video}: {cent:.3f} centrality ({self.graph.degree(video)} connections)")
            
        # Find strongest connections
        strongest_connections = []
        for u, v, data in self.graph.edges(data=True):
            strongest_connections.append((u, v, data['weight']))
        
        strongest_connections.sort(key=lambda x: x[2], reverse=True)
        
        print("\nStrongest Video Connections:")
        for v1, v2, weight in strongest_connections[:5]:
            print(f"{v1} - {v2}: {weight} co-occurrences")
            
        # Print clustering information
        clustering_coef = nx.average_clustering(self.graph)
        print(f"\nAverage clustering coefficient: {clustering_coef:.3f}")

def main():
    """Generate and save graph visualization"""
    try:
        # Initialize visualizer
        visualizer = UserSessionVisualizer()
        
        # Create and visualize graph
        # Start with a lower minimum weight to ensure we get some connections
        visualizer.create_graph(min_weight=5)  # Show connections occurring 5+ times
        visualizer.visualize()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 