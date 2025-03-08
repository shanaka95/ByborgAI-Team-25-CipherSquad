import os
import csv
import random
import pandas as pd
from typing import List, Dict, Any

# List of possible search query templates
SEARCH_QUERY_TEMPLATES = [
    "videos about {topic}",
    "show me {topic} videos",
    "find {topic} content",
    "{topic} tutorials",
    "best {topic} videos",
    "{topic} examples",
    "trending {topic} videos",
    "popular {topic} content",
    "recent {topic} uploads",
    "{topic} recommendations",
    "top {topic} videos",
    "{topic} highlights",
    "{topic} compilation",
    "learn about {topic}",
    "{topic} guide",
    "{topic} demonstration",
    "how to {topic}",
    "{topic} explained",
    "{topic} review",
    "{topic} analysis"
]

# List of possible topics for search queries
TOPICS = [
    "nature",
    "wildlife",
    "technology",
    "science",
    "history",
    "art",
    "music",
    "sports",
    "cooking",
    "travel",
    "adventure",
    "education",
    "gaming",
    "fashion",
    "health",
    "fitness",
    "meditation",
    "programming",
    "photography",
    "filmmaking",
    "animation",
    "design",
    "architecture",
    "engineering",
    "space",
    "astronomy",
    "physics",
    "chemistry",
    "biology",
    "mathematics",
    "economics",
    "politics",
    "philosophy",
    "psychology",
    "sociology",
    "anthropology",
    "archaeology",
    "geography",
    "geology",
    "oceanography",
    "meteorology",
    "environmental science",
    "renewable energy",
    "artificial intelligence",
    "machine learning",
    "data science",
    "cybersecurity",
    "blockchain",
    "virtual reality",
    "augmented reality"
]

def generate_random_search_query() -> str:
    """Generate a random search query using templates and topics"""
    template = random.choice(SEARCH_QUERY_TEMPLATES)
    topic = random.choice(TOPICS)
    return template.format(topic=topic)

def extract_user_ids_from_sessions(sessions_file: str) -> List[str]:
    """Extract unique user IDs from the user sessions CSV file"""
    user_ids = []
    
    try:
        # Read the CSV file
        df = pd.read_csv(sessions_file)
        
        # Extract user IDs
        user_ids = df['user_id'].unique().tolist()
        print(f"Extracted {len(user_ids)} unique user IDs from {sessions_file}")
        
    except Exception as e:
        print(f"Error extracting user IDs: {str(e)}")
        
    return user_ids

def generate_user_search_queries_csv(
    sessions_file: str, 
    output_file: str, 
    min_queries_per_user: int = 1,
    max_queries_per_user: int = 5
) -> None:
    """
    Generate a CSV file with user IDs and random search queries.
    
    Args:
        sessions_file: Path to the user sessions CSV file
        output_file: Path to the output CSV file
        min_queries_per_user: Minimum number of search queries per user
        max_queries_per_user: Maximum number of search queries per user
    """
    try:
        # Extract user IDs
        user_ids = extract_user_ids_from_sessions(sessions_file)
        
        if not user_ids:
            print("No user IDs found. Exiting.")
            return
        
        # Generate search queries for each user
        user_queries = []
        
        for user_id in user_ids:
            # Randomly determine how many queries this user will have
            num_queries = random.randint(min_queries_per_user, max_queries_per_user)
            
            # Generate the specified number of queries for this user
            for _ in range(num_queries):
                query = generate_random_search_query()
                user_queries.append({
                    'user_id': user_id,
                    'search_query': query
                })
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['user_id', 'search_query']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in user_queries:
                writer.writerow(row)
        
        print(f"Generated {len(user_queries)} search queries for {len(user_ids)} users")
        print(f"Output saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating user search queries: {str(e)}")

def main():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sessions_file = os.path.join(current_dir, "user_sessions.csv")
    output_file = os.path.join(current_dir, "user_search_queries.csv")
    
    # Generate user search queries
    generate_user_search_queries_csv(
        sessions_file=sessions_file,
        output_file=output_file,
        min_queries_per_user=1,
        max_queries_per_user=5
    )

if __name__ == "__main__":
    main() 