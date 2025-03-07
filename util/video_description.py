import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging
from typing import Dict, List
import os
from PIL import Image
import cv2
from tqdm import tqdm
from create_video_descriptions_collection import VideoDescriptionManager
import glob

logger = logging.getLogger(__name__)

class LightVideoDescriber:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize the lightweight video description model and Qdrant client"""
        try:
            # Set logging level
            logging.basicConfig(level=logging.INFO)
            
            # Initialize description manager
            self.description_manager = VideoDescriptionManager(qdrant_host, qdrant_port)
            logger.info("Connected to Qdrant database")
            
            # Check CUDA availability and set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load model with better error handling
            try:
                model_name = "Salesforce/blip2-opt-2.7b"  # Using BLIP-2 which is more reliable
                self.processor = Blip2Processor.from_pretrained(model_name)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load video description model: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error initializing video description model: {str(e)}")
            raise

    def process_all_videos(self, data_folder: str) -> List[Dict]:
        """
        Process all videos in the specified folder and update their descriptions in Qdrant.
        Returns a list of processing results.
        """
        try:
            # Ensure collections exist
            self.description_manager.setup_collections()
            
            # Get all video files
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(data_folder, f"*{ext}")))
            
            if not video_files:
                logger.warning(f"No video files found in {data_folder}")
                return []
            
            logger.info(f"Found {len(video_files)} video files to process")
            results = []
            
            # Process each video
            for video_path in tqdm(video_files, desc="Processing videos"):
                try:
                    # Get video description
                    result = self.describe_video(video_path)
                    
                    # Save description using the manager
                    video_name = os.path.basename(video_path)
                    description_id = self.description_manager.save_description(video_name, result)
                    
                    result["video_name"] = video_name
                    result["description_id"] = description_id
                    result["status"] = "success"
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {str(e)}")
                    results.append({
                        "video_name": os.path.basename(video_path),
                        "status": "error",
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch video processing: {str(e)}")
            raise

    def extract_frames(self, video_path: str, num_frames: int = 5) -> list:
        """Extract evenly spaced frames from the video"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")
                
            frame_indices = list(range(0, total_frames, total_frames // num_frames))[:num_frames]
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    frame_pil = Image.fromarray(frame_rgb)
                    frames.append(frame_pil)
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames could be extracted from the video")
                
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

    def describe_video(self, video_path: str) -> Dict[str, str]:
        """
        Generate a description of the video content using the lightweight model.
        Returns a dictionary with the description and metadata.
        """
        try:
            # Extract frames
            logger.info("Extracting frames from video...")
            frames = self.extract_frames(video_path)
            
            # Process each frame and generate descriptions
            logger.info("Generating descriptions for frames...")
            descriptions = []
            
            for frame in tqdm(frames, desc="Processing frames"):
                # Prepare the image
                inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
                
                # Generate description
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode the output
                description = self.processor.decode(outputs[0], skip_special_tokens=True)
                if description:  # Only add non-empty descriptions
                    descriptions.append(description)
            
            if not descriptions:
                raise ValueError("No descriptions could be generated for the video frames")
                
            # Combine descriptions into a coherent summary
            combined_description = self._combine_descriptions(descriptions)
            
            result = {
                "description": combined_description,
                "num_frames_analyzed": len(frames),
                "model_used": "BLIP-2 OPT-2.7B"
            }
            
            logger.info("Video description completed")
            return result
            
        except Exception as e:
            logger.error(f"Error describing video: {str(e)}")
            raise

    def _combine_descriptions(self, descriptions: list) -> str:
        """Combine multiple frame descriptions into a coherent summary"""
        # Remove duplicates while preserving order
        unique_descriptions = []
        seen = set()
        for desc in descriptions:
            if desc.lower() not in seen:
                unique_descriptions.append(desc)
                seen.add(desc.lower())
        
        # Join descriptions with proper transitions
        if len(unique_descriptions) == 1:
            return unique_descriptions[0]
        
        combined = "The video shows " + unique_descriptions[0].lower()
        for desc in unique_descriptions[1:-1]:
            combined += f". Then, {desc.lower()}"
        if len(unique_descriptions) > 1:
            combined += f". Finally, {unique_descriptions[-1].lower()}"
        
        return combined

def main():
    """
    Example usage of the VideoDescriber for batch processing
    """
    try:
        # Initialize the describer
        describer = LightVideoDescriber()
        
        # Process all videos in the data folder
        data_folder = "../data/"
        results = describer.process_all_videos(data_folder)
        
        # Print results summary
        print("\nProcessing Results Summary:")
        print(f"Total videos processed: {len(results)}")
        
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        # Print details of failed videos
        failed = [r for r in results if r["status"] == "error"]
        if failed:
            print("\nFailed Videos:")
            for result in failed:
                print(f"- {result['video_name']}: {result['error']}")
        
        # Print successful descriptions
        successful_videos = [r for r in results if r["status"] == "success"]
        if successful_videos:
            print("\nSuccessful Descriptions:")
            for result in successful_videos:
                print(f"\nVideo: {result['video_name']}")
                print(f"Description ID: {result['description_id']}")
                print(f"Description: {result['description']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 