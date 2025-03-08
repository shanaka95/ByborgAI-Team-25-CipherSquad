import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file, make_response, stream_with_context
import pandas as pd
import json
import numpy as np
import mimetypes
import re
import subprocess
import tempfile
import shutil
import time
import threading
from pathlib import Path

# Add parent directory to path to import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from recommender import VideoRecommender
except ImportError:
    from mock_recommender import VideoRecommender

from prompt import generate_preference_summary_prompt
from llm import generate_preference_summary
try:
    from find_video_segment import find_best_matching_segment
except ImportError:
    # Create a dummy function if the real one is not available
    def find_best_matching_segment(text, video_name, window_size=5):
        return None, None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize the recommender
recommender = VideoRecommender()

# Add path to converted videos directory
CONVERTED_VIDEOS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_converted")

# Create the directory if it doesn't exist (in case batch conversion hasn't been run yet)
os.makedirs(CONVERTED_VIDEOS_DIR, exist_ok=True)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_user_sessions', methods=['GET'])
def get_user_sessions():
    """Get all user sessions from the CSV file"""
    try:
        # Path to user sessions CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sessions_path = os.path.join(parent_dir, "user-sessions", "user_sessions.csv")
        
        # Read the CSV file
        df = pd.read_csv(sessions_path)
        
        # Replace NaN values with None (which will be converted to null in JSON)
        df = df.replace({np.nan: None})
        
        # Convert to list of dictionaries
        users = df.to_dict(orient='records')
        
        return jsonify({"success": True, "users": users})
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations for a user"""
    try:
        data = request.json
        user_id = data.get('user_id')
        session_videos = data.get('session_videos', [])
        
        # Filter out empty videos
        session_videos = [video for video in session_videos if video]
        
        if not user_id or not session_videos:
            return jsonify({"success": False, "error": "User ID and session videos are required"})
        
        # Get recommendations
        recommendations = recommender.get_recommendations_as_list(user_id, session_videos)
        
        # Get video descriptions and find best matching segments
        video_descriptions = {}
        video_segments = {}
        user_queries_text = ""
        
        # Get user search queries first to use in segment matching
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        queries_path = os.path.join(parent_dir, "user-sessions", "user_search_queries.csv")
        
        # Read the CSV file
        df = pd.read_csv(queries_path)
        
        # Filter by user_id
        user_queries = df[df['user_id'] == user_id]['search_query'].tolist()
        
        # Join all user queries to use for segment matching
        if user_queries:
            user_queries_text = " ".join(user_queries)
        
        for video_id in recommendations:
            video_data = recommender.get_video_description(video_id)
            if "error" not in video_data:
                description = video_data.get("description", "No description available")
                video_descriptions[video_id] = description
                
                # Try to find the best matching segment using user queries
                try:
                    if user_queries_text:
                        start_frame, end_frame = find_best_matching_segment(
                            user_queries_text, 
                            video_id,
                            window_size=5
                        )
                        if start_frame is not None and end_frame is not None:
                            # The frames returned are actually seconds in this implementation
                            video_segments[video_id] = {
                                "start_frame": start_frame,
                                "end_frame": end_frame
                            }
                            # Add segment info to description - use "seconds" instead of "frames"
                            video_descriptions[video_id] = f"{description} [Best matching segment: seconds {start_frame}-{end_frame}]"
                except Exception as segment_error:
                    logger.error(f"Error finding best segment for {video_id}: {str(segment_error)}")
            else:
                video_descriptions[video_id] = "No description available"
        
        # Create combined text
        combined_text = "\nRECOMMENDED VIDEOS DESCRIPTIONS:\n"
        for video_id, description in video_descriptions.items():
            combined_text += f"Video {video_id}: {description}\n\n"
        
        combined_text += "USER SEARCH QUERIES:\n"
        for query in user_queries:
            combined_text += f"- {query}\n"
        
        # Generate preference summary prompt
        preference_prompt = generate_preference_summary_prompt(combined_text)
        
        # Generate preference summary
        preference_summary = generate_preference_summary(preference_prompt)
        
        return jsonify({
            "success": True, 
            "recommendations": recommendations,
            "video_descriptions": video_descriptions,
            "video_segments": video_segments,
            "user_queries": user_queries,
            "preference_summary": preference_summary
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/video/<video_id>', methods=['GET'])
def serve_video(video_id):
    """Serve a video file from the data directory with proper headers for streaming"""
    try:
        # Path to data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, "data")
        
        # Check if format parameter is provided
        requested_format = request.args.get('format', 'original').lower()
        
        # Handle different video ID formats
        # Replace 'sceneclipautoautotrain' with 'scenecliptest' if present
        if 'sceneclipautoautotrain' in video_id:
            video_id = video_id.replace('sceneclipautoautotrain', 'scenecliptest')
            logger.info(f"Converted video ID to: {video_id}")
        
        # Extract the base video ID (without extension)
        if video_id.endswith('.avi'):
            base_video_id = video_id[:-4]
        else:
            base_video_id = video_id
            
        # Remove any scenecliptest prefix for the base ID
        if base_video_id.startswith('scenecliptest'):
            clean_video_id = base_video_id[len('scenecliptest'):]
        else:
            clean_video_id = base_video_id
            
        # Check if video_id already includes the prefix
        if not base_video_id.startswith('scenecliptest'):
            # If not, add the prefix and ensure it's in the correct format
            # Try to convert to integer and format with leading zeros
            try:
                video_num = int(clean_video_id)
                video_filename = f"scenecliptest{video_num:05d}.avi"
                mp4_filename = f"scenecliptest{video_num:05d}.mp4"
            except ValueError:
                # If conversion fails, use the ID as is
                video_filename = f"scenecliptest{clean_video_id}.avi"
                mp4_filename = f"scenecliptest{clean_video_id}.mp4"
        else:
            # If it already has the prefix, just add the extension if needed
            video_filename = base_video_id + '.avi' if not base_video_id.endswith('.avi') else base_video_id
            mp4_filename = base_video_id + '.mp4' if not base_video_id.endswith('.mp4') else base_video_id
        
        # Check for a pre-converted MP4 file first if MP4 format is requested
        if requested_format == 'mp4':
            converted_path = os.path.join(CONVERTED_VIDEOS_DIR, mp4_filename)
            logger.info(f"Checking for converted file at: {converted_path}")
            
            if os.path.exists(converted_path):
                logger.info(f"Found pre-converted MP4 file: {converted_path}")
                
                # Handle range requests for the MP4 file
                file_size = os.path.getsize(converted_path)
                range_header = request.headers.get('Range', None)
                
                if range_header:
                    m = re.search('bytes=(?P<start>\d+)-(?P<end>\d+)?', range_header)
                    if m:
                        start = int(m.group('start'))
                        end = int(m.group('end')) if m.group('end') else file_size - 1
                        
                        # Ensure end isn't beyond the file size
                        end = min(end, file_size - 1)
                        
                        # Calculate response size
                        response_size = end - start + 1
                        
                        # Create a response with the requested range
                        response = Response(
                            stream_with_context(_partial_file_reader(converted_path, start, end)),
                            status=206,
                            mimetype='video/mp4',
                            content_type='video/mp4',
                            direct_passthrough=True
                        )
                        
                        # Set Content-Range and other headers
                        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
                        response.headers.add('Accept-Ranges', 'bytes')
                        response.headers.add('Content-Length', str(response_size))
                        response.headers.add('Content-Disposition', f'inline; filename={mp4_filename}')
                        
                        return response
                
                # If no range header, serve the whole file
                return send_file(
                    converted_path,
                    mimetype='video/mp4',
                    as_attachment=False,
                    conditional=True
                )
            else:
                logger.warning(f"No pre-converted MP4 file found at: {converted_path}")
                # Fall back to original file
        
        # Path to original video file
        video_path = os.path.join(data_dir, video_filename)
        logger.info(f"Attempting to serve original video: {video_path}")
        
        # Final check if the file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return jsonify({"success": False, "error": "Video file not found"}), 404
        
        # Handle range requests for video streaming
        file_size = os.path.getsize(video_path)
        range_header = request.headers.get('Range', None)
        
        if range_header:
            m = re.search('bytes=(?P<start>\d+)-(?P<end>\d+)?', range_header)
            if m:
                start = int(m.group('start'))
                end = int(m.group('end')) if m.group('end') else file_size - 1
                
                # Ensure end isn't beyond the file size
                end = min(end, file_size - 1)
                
                # Calculate response size
                response_size = end - start + 1
                
                # Create a response with the requested range
                response = Response(
                    stream_with_context(_partial_file_reader(video_path, start, end)),
                    status=206,
                    mimetype='video/x-msvideo',
                    content_type='video/x-msvideo',
                    direct_passthrough=True
                )
                
                # Set Content-Range and other headers
                response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
                response.headers.add('Accept-Ranges', 'bytes')
                response.headers.add('Content-Length', str(response_size))
                response.headers.add('Content-Disposition', f'inline; filename={video_filename}')
                
                return response
        
        # For normal requests, send the whole file
        response = make_response(send_file(
            video_path,
            mimetype='video/x-msvideo', 
            as_attachment=False,
            download_name=video_filename,
            conditional=True
        ))
        
        # Get video file size
        file_size = os.path.getsize(video_path)
        
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Content-Disposition', f'inline; filename={video_filename}')
        
        return response
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
        
def _partial_file_reader(file_path, start, end):
    """Helper function to read a file in chunks for range requests"""
    with open(file_path, 'rb') as f:
        f.seek(start)
        remaining = end - start + 1
        chunk_size = 8192  # 8KB chunks
        
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

@app.route('/video_player/<video_id>', methods=['GET'])
def video_player(video_id):
    """Show a dedicated video player page that can handle AVI files better"""
    try:
        # We'll create a simple page with video controls
        video_url = f'/video/{video_id}'
        
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Player</title>
            <style>
                body {{ margin: 0; padding: 0; overflow: hidden; background-color: #000; }}
                .video-container {{ width: 100%; height: 100vh; display: flex; justify-content: center; align-items: center; }}
                video {{ max-width: 100%; max-height: 100%; }}
                .fallback {{ color: white; text-align: center; padding: 20px; }}
                .fallback a {{ color: #4a6bff; }}
            </style>
        </head>
        <body>
            <div class="video-container">
                <video controls autoplay>
                    <source src="{video_url}" type="video/x-msvideo">
                    <div class="fallback">
                        <p>Your browser doesn't support this video format.</p>
                        <p><a href="{video_url}" download>Download the video</a> to watch it on your device.</p>
                    </div>
                </video>
            </div>
            <script>
                // Add error handling
                document.querySelector('video').addEventListener('error', function() {{
                    document.querySelector('.video-container').innerHTML = `
                        <div class="fallback">
                            <p>Sorry, there was an error playing this video.</p>
                            <p><a href="{video_url}" download>Download the video</a> to watch it on your device.</p>
                        </div>
                    `;
                }});
            </script>
        </body>
        </html>
        '''
        
        return html
    except Exception as e:
        logger.error(f"Error creating video player: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 