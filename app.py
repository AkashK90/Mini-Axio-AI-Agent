"""
Flask Web UI for Video Similarity Finder
Simple interface to scan YouTube videos and find similar content
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from video_similarity_finder import VideoSimilarityFinder, SimilarityResult
#from video import VideoSimilarityFinder, SimilarityResult
from typing import List
import logging

#ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the similarity finder
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

finder = None
if YOUTUBE_API_KEY:
    finder = VideoSimilarityFinder(YOUTUBE_API_KEY)
else:
    logger.warning("YouTube API key not set. Please set YOUTUBE_API_KEY environment variable.")

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def scan_video():
    """API endpoint to scan a video for similarities"""
    try:
        data = request.get_json()
        video_url = data.get('url', '').strip()
        top_k = data.get('top_k', 5)
        
        if not video_url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if not finder:
            return jsonify({'error': 'System not configured. Please set API keys.'}), 500
        
        logger.info(f"Scanning video: {video_url}")       
        # Find similar videos
        results = finder.find_similar(video_url, top_k=top_k)
        
        # Format results for frontend
        formatted_results = [
            {
                'title': r.title,
                'url': r.video_url,
                'video_id': r.video_id,
                'similarity': round(r.similarity_score * 100, 1),
                'classification': r.classification,
                'reason': r.reason,
                'thumbnail': f"https://img.youtube.com/vi/{r.video_id}/mqdefault.jpg"
            }
            for r in results
        ]
        
        #explanation
        explanation = finder.agent.explain_similarity(results)
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'explanation': explanation,
            'source_video': video_url
        })
    
    except Exception as e:
        logger.error(f"Error scanning video: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'configured': finder is not None
    })
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)