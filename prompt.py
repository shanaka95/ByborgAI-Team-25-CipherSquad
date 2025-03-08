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
    Generate a prompt for few-shot learning to summarize user preferences in a single sentence.
    
    Args:
        combined_text: Combined text of product descriptions and user search queries
        
    Returns:
        A prompt for the LLM to summarize user preferences in one sentence
    """
    logger.info("Generating one-sentence preference summary prompt")
    
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
            "summary": "This user is primarily interested in data science and machine learning content, especially educational tutorials that provide both theoretical knowledge and practical implementation examples."
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
            "summary": "This user has a strong preference for high-quality nature and wildlife content with immersive visuals and relaxing qualities."
        }
    ]
    
    # Construct the prompt with few-shot examples
    prompt = "You are an expert at analyzing user preferences based on their video recommendations and search queries. Your task is to summarize what the user commonly wants to see in videos in a SINGLE SENTENCE.\n\n"
    
    # Add the few-shot examples
    for i, example in enumerate(examples, 1):
        prompt += f"EXAMPLE {i}:\n"
        prompt += f"Combined Text:\n{example['combined_text']}\n\n"
        prompt += f"One-Sentence Summary:\n{example['summary']}\n\n"
        prompt += "-" * 80 + "\n\n"
    
    # Add the current user's data and instructions
    prompt += "Now, analyze the following user's data:\n\n"
    prompt += f"Combined Text:\n{combined_text}\n\n"
    prompt += "One-Sentence Summary:\n"
    prompt += "Provide a SINGLE SENTENCE that captures this user's main video preferences and interests. Be concise but comprehensive."
    
    logger.info("One-sentence preference summary prompt generated successfully")
    return prompt
