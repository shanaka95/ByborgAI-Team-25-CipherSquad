# Video Recommendation System

## Overview
This project develops a video recommendation and generation system that offers personalized video suggestions based on user preferences, trending content, and recent uploads. Also, the system generates short video previews, selects the most engaging thumbnails, and creates relevant captions based on user behavior to enhance the overall user experience.

## Features
- **Video Dataset**: We used the Hollywood Movies dataset (https://www.di.ens.fr/~laptev/actions/hollywood2/), which contains over 1,000 short movie clips, totaling approximately 26GB of data.
- **Video Recommendations**: We use the following techniques to filter and recommend 20-30 videos that match user behavior.

   - Vector Similarity: We use the BERT model to compute the cosine distance between text embeddings of video descriptions, helping identify similar videos.
   - Node2Vec Model trained on user session data
   - Currently popular and trending hit videos
   - Recently uploaded videos
- Re-Ranking with LTR (Learning to Rank): Prioritizing the selected videos using an LTR algorithm to generate the top 10 recommendations.
- Content Generation:
  - We generate a query for each user describing their preferences based on their watch history and likes.
  - We use a custom avarage distance based algorithm with clip model identify the best video segmentation for that user.
  - We generate short, engaging preview videos depending on the video segment.
  - Thumbnails customized to user preferences.

## Workflow
1. **Web app**: Use our Flas web app to test the whole process which is located inside webapp folder.

 

## Technologies Used
- **Programming Language**: Python, JavaScript
- **Machine Learning**: LambdaMART, Node2Vec, BERT, Clip, LLaMA
- **Web Framework**: Flask
- **Database**: SQLite, QDRANT
- **Frontend**: React, TailwindCSS
- **Backend**: Python


## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/video-recommendation.git
   ```
2. Navigate to the project directory:
   ```sh
   cd webapp
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   python app.py
   ```

## Usage
1. Select User sessions from the dropdown menu.
2. View recommendations.
3. Enjoy personalized video previews, captions, and thumbnails tailored to users interests.


