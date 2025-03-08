import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_preference_summary_prompt(combined_text: str) -> str:
    """
    Generate a prompt for few-shot learning to summarize user preferences.
    
    Args:
        combined_text: Combined text of product descriptions and user search queries
        
    Returns:
        A prompt for the LLM to summarize user preferences
    """
    logger.info("Generating preference summary prompt")
    
    # Create the few-shot examples
    examples = [
        {
            "combined_text": """
RECOMMENDED VIDEOS DESCRIPTIONS:
Video video123: This video explores the fundamentals of machine learning algorithms and their applications in real-world scenarios.
Video video456: A comprehensive tutorial on Python programming for data science, covering pandas, numpy, and matplotlib.
Video video789: Deep dive into neural networks architecture and how they can be implemented using TensorFlow.

USER SEARCH QUERIES:
- machine learning tutorials
- python for data science
- neural networks explained
- tensorflow examples
            """,
            "summary": """
Based on the recommended videos and search queries, this user shows a strong interest in data science and machine learning. They are particularly focused on:

1. Machine learning fundamentals and algorithms
2. Python programming for data analysis
3. Neural networks and deep learning
4. Practical implementation using frameworks like TensorFlow

The user appears to be someone who is likely studying or working in the field of data science or AI, looking for both theoretical knowledge and practical implementation examples. They prefer educational content that provides clear explanations and tutorials.

For future recommendations, this user would likely appreciate advanced machine learning techniques, specialized deep learning architectures, and practical projects that demonstrate real-world applications of these technologies.
            """
        },
        {
            "combined_text": """
RECOMMENDED VIDEOS DESCRIPTIONS:
Video video321: Beautiful landscapes from around the world captured in 4K resolution with ambient music.
Video video654: Wildlife documentary featuring rare animals in their natural habitats across Africa.
Video video987: Underwater exploration of coral reefs and marine life in the Great Barrier Reef.

USER SEARCH QUERIES:
- beautiful nature videos
- wildlife documentaries
- underwater exploration
- relaxing landscape videos
- 4K nature footage
            """,
            "summary": """
Based on the recommended videos and search queries, this user has a clear passion for nature and wildlife content. Their preferences include:

1. High-quality (4K) landscape videos with aesthetic appeal
2. Wildlife documentaries focusing on animals in their natural habitats
3. Underwater and marine life exploration
4. Content with a relaxing, ambient quality

The user appears to enjoy immersive, visually stunning content that showcases the natural world. They likely use these videos both for entertainment and relaxation purposes.

For future recommendations, this user would appreciate content featuring unexplored natural locations, rare wildlife footage, seasonal nature changes, and perhaps nature conservation topics. They would likely prefer videos with high production quality and immersive audio.
            """
        }
    ]
    
    # Construct the prompt with few-shot examples
    prompt = "You are an expert at analyzing user preferences based on their video recommendations and search queries. Your task is to summarize what the user commonly wants to see in videos and identify their preferences.\n\n"
    
    # Add the few-shot examples
    for i, example in enumerate(examples, 1):
        prompt += f"EXAMPLE {i}:\n"
        prompt += f"Combined Text:\n{example['combined_text']}\n\n"
        prompt += f"Preference Summary:\n{example['summary']}\n\n"
        prompt += "-" * 80 + "\n\n"
    
    # Add the current user's data and instructions
    prompt += "Now, analyze the following user's data:\n\n"
    prompt += f"Combined Text:\n{combined_text}\n\n"
    prompt += "Preference Summary:\n"
    prompt += "Based on the recommended videos and search queries, provide a detailed summary of this user's preferences. Include:\n"
    prompt += "1. The main topics and themes they're interested in\n"
    prompt += "2. The type of content they prefer (tutorials, entertainment, etc.)\n"
    prompt += "3. Any specific qualities they look for in videos\n"
    prompt += "4. Suggestions for what kind of content would appeal to them in the future\n\n"
    prompt += "Your analysis should be comprehensive but concise, focusing on identifying patterns in their behavior and preferences."
    
    logger.info("Preference summary prompt generated successfully")
    return prompt
