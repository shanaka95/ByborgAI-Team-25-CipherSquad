import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import numpy as np

# Add parent directory to path to import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from recommender import VideoRecommender
except ImportError:
    from mock_recommender import VideoRecommender

from prompt import generate_preference_summary_prompt
from llm import generate_preference_summary

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
        
        # Get video descriptions
        video_descriptions = {}
        for video_id in recommendations:
            video_data = recommender.get_video_description(video_id)
            if "error" not in video_data:
                video_descriptions[video_id] = video_data.get("description", "No description available")
            else:
                video_descriptions[video_id] = "No description available"
        
        # Get user search queries
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        queries_path = os.path.join(parent_dir, "user-sessions", "user_search_queries.csv")
        
        # Read the CSV file
        df = pd.read_csv(queries_path)
        
        # Filter by user_id
        user_queries = df[df['user_id'] == user_id]['search_query'].tolist()
        
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
            "user_queries": user_queries,
            "preference_summary": preference_summary
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 