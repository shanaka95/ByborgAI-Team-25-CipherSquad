import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
from typing import Dict, Tuple
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InteractiveVideoVisualizer:
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
            expected_cols = ['user_id'] + [f'video{i}' for i in range(1, 5)]
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
        video_stats = defaultdict(int)  # Track individual video occurrences
        
        # Get video columns
        video_cols = [col for col in self.df.columns if col.startswith('video')]
        
        # Count co-occurrences of videos
        logger.info("Analyzing video connections...")
        total_connections = 0
        
        for _, row in self.df.iterrows():
            # Get non-empty videos for this user
            videos = [v for v in row[video_cols] if pd.notna(v) and v is not None]
            
            # Count individual video occurrences
            for video in videos:
                video_stats[video] += 1
            
            # Count co-occurrences
            for i in range(len(videos)):
                for j in range(i + 1, len(videos)):
                    # Sort video names to ensure consistent edge keys
                    v1, v2 = sorted([videos[i], videos[j]])
                    connections[(v1, v2)] += 1
                    total_connections += 1
                    
        self.video_stats = video_stats
        logger.info(f"Found {total_connections} total connections between {len(video_stats)} unique videos")
        return connections
        
    def create_graph(self, min_weight: int = 2):
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
                    
            # Add node attributes
            max_occurrences = max(self.video_stats.values())
            for node in self.graph.nodes():
                occurrences = self.video_stats[node]
                # Normalize size between 20 and 50
                size = 20 + 30 * (occurrences / max_occurrences)
                self.graph.nodes[node]['size'] = size
                self.graph.nodes[node]['occurrences'] = occurrences
                
            logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise
            
    def create_interactive_visualization(self, output_file: str = "video_network.html"):
        """Create interactive HTML visualization"""
        try:
            if self.graph.number_of_nodes() == 0:
                raise ValueError("Graph is empty. Check if create_graph() was successful.")
                
            logger.info("Generating interactive visualization...")
            
            # Create Pyvis network
            net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
            net.force_atlas_2based()
            
            # Calculate node colors based on occurrences
            max_occurrences = max(nx.get_node_attributes(self.graph, 'occurrences').values())
            
            # Add nodes with size and color based on occurrences
            for node in self.graph.nodes():
                occurrences = self.graph.nodes[node]['occurrences']
                size = self.graph.nodes[node]['size']
                # Color gradient from light blue to dark blue based on occurrences
                color_intensity = int(50 + 205 * (occurrences / max_occurrences))
                color = f"#{color_intensity:02x}{'80':>2}{'ff':>2}"
                
                net.add_node(
                    node,
                    label=node.split('.')[0],  # Remove file extension
                    title=f"Video: {node}\\nViews: {occurrences}",
                    size=size,
                    color=color
                )
            
            # Add edges with tooltips and varying thickness
            max_weight = max(d['weight'] for _, _, d in self.graph.edges(data=True))
            for u, v, data in self.graph.edges(data=True):
                # Edge thickness based on weight
                width = 1 + 9 * (data['weight'] / max_weight)  # Scale from 1 to 10
                net.add_edge(
                    u, v,
                    value=width,
                    title=f"Co-views: {data['weight']}",
                    arrowStrengthing=False
                )
            
            # Configure physics and interaction
            net.set_options("""
            var options = {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08,
                        "damping": 0.4
                    },
                    "maxVelocity": 50,
                    "minVelocity": 0.1,
                    "solver": "forceAtlas2Based"
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 100,
                    "zoomView": true,
                    "dragView": true,
                    "dragNodes": true,
                    "hideEdgesOnDrag": true
                },
                "edges": {
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    },
                    "color": {"inherit": "both"}
                }
            }
            """)
            
            # Save the visualization
            output_path = os.path.join("../data", output_file)
            net.save_graph(output_path)
            logger.info(f"Interactive visualization saved to {output_path}")
            
            # Generate statistics file
            self._save_statistics(output_path.replace('.html', '_stats.json'))
            
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            raise
            
    def _save_statistics(self, output_file: str):
        """Save graph statistics to JSON file"""
        try:
            # Calculate statistics
            stats = {
                "graph_stats": {
                    "num_nodes": self.graph.number_of_nodes(),
                    "num_edges": self.graph.number_of_edges(),
                    "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                    "density": nx.density(self.graph),
                    "avg_clustering": nx.average_clustering(self.graph)
                },
                "video_stats": {
                    "total_videos": len(self.video_stats),
                    "total_views": sum(self.video_stats.values()),
                    "avg_views_per_video": sum(self.video_stats.values()) / len(self.video_stats),
                    "top_videos": sorted(
                        [{"video": v, "views": c} for v, c in self.video_stats.items()],
                        key=lambda x: x["views"],
                        reverse=True
                    )[:10]
                },
                "connection_stats": {
                    "strongest_connections": [
                        {
                            "video1": u,
                            "video2": v,
                            "co_views": d["weight"]
                        }
                        for u, v, d in sorted(
                            self.graph.edges(data=True),
                            key=lambda x: x[2]["weight"],
                            reverse=True
                        )[:10]
                    ]
                }
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {str(e)}")
            raise

def main():
    """Generate interactive visualization"""
    try:
        # Initialize visualizer
        visualizer = InteractiveVideoVisualizer()
        
        # Create graph and visualization
        visualizer.create_graph(min_weight=2)  # Show connections occurring 2+ times
        visualizer.create_interactive_visualization()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 