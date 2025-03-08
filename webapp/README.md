# ByborgAI Video Recommendations Web Application

A modern web interface for displaying refined video recommendations based on user preferences.

## Features

- User selection from existing user sessions
- Display of watched videos for selected users
- Display of user search queries
- Generation of personalized video recommendations
- One-sentence summary of user preferences using LLM
- Responsive design for all device sizes

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the parent directory is in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ByborgAI
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

3. Select a user from the dropdown menu
4. View the user's watched videos
5. Click "Get Recommendations" to see personalized video suggestions
6. Click on any video card to see more details

## Architecture

The web application consists of:

- **Flask Backend**: Handles API requests and integrates with the recommendation system
- **Modern UI**: Built with Bootstrap 5 for responsive design
- **JavaScript Frontend**: Manages user interactions and dynamic content updates
- **Integration with LLM**: Generates concise preference summaries

## Dependencies

- Flask: Web framework
- Flask-Session: Server-side session management
- Pandas: Data manipulation
- Requests: HTTP requests
- Hugging Face Hub: LLM integration

- Hugging Face Hub: LLM integration 