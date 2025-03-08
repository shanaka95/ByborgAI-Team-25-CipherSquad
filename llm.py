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
        return False
