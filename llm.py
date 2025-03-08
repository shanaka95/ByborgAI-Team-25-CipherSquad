try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_preference_summary(prompt):
    """
    Generate a user preference summary based on the provided prompt using Hugging Face's LLM.
    
    Args:
        prompt: The prompt containing user data and instructions
        
    Returns:
        A summary of user preferences
    """
    if not HF_AVAILABLE:
        logger.warning("Hugging Face client not available. Using mock response.")
        return _generate_mock_response(prompt)
    
    try:
        logger.info("Generating preference summary using Hugging Face LLM")
        
        try:
            # Try with provider parameter
            client = InferenceClient(provider="nebius")
        except TypeError:
            # Fall back to default initialization if provider is not supported
            logger.warning("Provider parameter not supported. Using default InferenceClient.")
            client = InferenceClient()

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct", 
                messages=messages, 
                max_tokens=1000,
            )
            
            content = completion.choices[0].message.content
            logger.info("Successfully generated preference summary")
            return content
        except AttributeError:
            # Fall back to older API if needed
            logger.warning("Using alternative API for Hugging Face client")
            response = client.text_generation(
                prompt=prompt,
                model="meta-llama/Llama-3.3-70B-Instruct",
                max_new_tokens=1000,
            )
            logger.info("Successfully generated preference summary")
            return response
    
    except Exception as e:
        logger.error(f"Error generating preference summary: {str(e)}")
        return _generate_mock_response(prompt)

def _generate_mock_response(prompt: str) -> str:
    """
    Generate a mock response when the API is not available.
    
    Args:
        prompt: The prompt that would have been sent to the API
        
    Returns:
        A mock response
    """
    logger.info("Generating mock response for preference summary")
    
    # Check if the prompt is for user preference analysis
    if "analyze the following user's data" in prompt.lower():
        return """
Based on the recommended videos and search queries, this user shows a diverse range of interests with a particular focus on technology, science, and educational content. Their preferences include:

1. Main topics and themes:
   - Technology (smartphones, tech reviews)
   - Space and astronomy (James Webb telescope, constellations)
   - Travel and cultural exploration (Southeast Asian countries, museums)
   - DIY and home improvement (kitchen renovation)
   - Wildlife and nature documentaries
   - Arts and music (symphony orchestra, classical music)
   - Extreme sports and adventure

2. Content type preferences:
   - Educational guides and tutorials (DIY, astronomy)
   - Documentary-style content (wildlife, space exploration)
   - Reviews and analysis (technology products)
   - Virtual tours and travel vlogs
   - Performance recordings (music)
   - Action-oriented compilations (extreme sports)

3. Specific qualities they look for:
   - Informative content with practical applications
   - Visual appeal and high production quality
   - Diverse subject matter across multiple domains
   - Content that combines entertainment with educational value
   - Expert commentary and guidance

4. Suggestions for future content:
   - In-depth technology tutorials and comparisons
   - More space and astronomy content, especially new discoveries
   - Advanced DIY projects beyond kitchen renovations
   - Immersive travel experiences to less-explored destinations
   - Documentaries that combine technology with other fields (tech in wildlife conservation, space technology applications)
   - Interactive guides that allow for deeper engagement with the subject matter

This user appears to be intellectually curious with wide-ranging interests, valuing content that is both entertaining and informative. They likely appreciate content that provides expert insights and practical knowledge they can apply in their own life, whether it's understanding technology, exploring the night sky, or planning travel adventures.
"""
    else:
        return "Mock response: Content generation is not available at the moment."
