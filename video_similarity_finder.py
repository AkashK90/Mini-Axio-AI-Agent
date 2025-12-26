"""
YouTube Video Similarity Finder
A system to find and explain similar videos using embeddings and LLM reasoning
"""
import os
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import yt_dlp
from yt_dlp.utils import DownloadError
from PIL import Image
import io
import hashlib
import json
# For embeddings, you'll need: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
# For vision: pip install torch torchvision timm
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
# For YouTube API: pip install google-api-python-client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

@dataclass
class VideoFeatures:
    """Container for extracted video features"""
    video_id: str
    url: str
    title: str
    description: str
    keyframes: List[np.ndarray]
    audio_fingerprint: str
    visual_embedding: np.ndarray
    text_embedding: np.ndarray
    metadata: Dict

@dataclass
class SimilarityResult:
    """Container for similarity search results"""
    video_url: str
    video_id: str
    title: str
    similarity_score: float
    reason: str
    classification: str  # 're-upload', 'edited-copy', 'unrelated', 'similar-content'


class VideoProcessor:
    """Handles video downloading and feature extraction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Visual model for keyframe embeddings
        self.visual_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_model = torch.nn.Sequential(*list(self.visual_model.children())[:-1])
        self.visual_model.eval()
        self.visual_model.to(self.device)
        
        # Text embedding model
        self.text_model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'shorts/' in url:
            return url.split('shorts/')[1].split('?')[0]
        elif 'watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return url
    
    def download_video_info(self, url: str) -> Dict:
        """Download video metadata and extract keyframes"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            #'cookiesfrombrowser':('chrome',),
            #'cookies':('cookies.txt',),
            "noplaylist": False, # true for no playlist
        }
        
        # try:
        #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #         info = ydl.extract_info(url, download=False)
        # except DownloadError:
        #     return {
        #         "error": "YouTube access blocked or cookies unavailable"
        #     }, 403
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)# set False if you only want metadata
            
        return info
        
        
        # try:
        #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #         info = ydl.extract_info(url, download=False)# set False if you only want metadata
            
        #     return info
        # except DownloadError as e:
        #     raise Exception("YouTube blocked the request. Login / bot verification required.") from e
            
    
    def extract_keyframes(self, url: str, n_frames: int = 7) -> List[Image.Image]:
        """Extract N evenly-spaced frames from video"""
        ydl_opts = {
            'format': 'worst',  # Download lowest quality for speed
            'quiet': True,
            'no_warnings': True,
        }
        
        # Note: For production, use opencv-python to extract frames
        # This is a simplified version
        frames = []
        
        # Placeholder: In real implementation, use cv2.VideoCapture
        # to extract frames at specific timestamps
        
        return frames
    
    def compute_visual_embedding(self, frames: List[Image.Image]) -> np.ndarray:
        """Compute visual embedding from keyframes"""
        if not frames:
            return np.zeros(2048)
        
        embeddings = []
        
        with torch.no_grad():
            for frame in frames:
                img_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
                embedding = self.visual_model(img_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        # Average pooling of frame embeddings
        return np.mean(embeddings, axis=0)
    
    def compute_audio_fingerprint(self, url: str) -> str:
        """Create simple audio fingerprint"""
        # Simplified: In production, use libraries like chromaprint or dejavu
        # For now, we'll create a hash based on audio properties
        
        ydl_opts = {
            'format': 'bestaudio',
            'quiet': True,
            'no_warnings': True,
        }
        
        # Placeholder fingerprint
        fingerprint = hashlib.md5(url.encode()).hexdigest()
        return fingerprint
    
    def process_video(self, url: str) -> VideoFeatures:
        """Main processing pipeline"""
        video_id = self.extract_video_id(url)
        info = self.download_video_info(url)
   
        # Extract features
        title = info.get('title', '')
        description = info.get('description', '')
        
        # For MVP, we'll use metadata-based embeddings
        # In production, extract actual frames
        keyframes = []  # self.extract_keyframes(url)
        
        # Compute embeddings
        text_content = f"{title}. {description}"
        text_embedding = self.text_model.encode(text_content)
        
        # Visual embedding (placeholder if no frames)
        visual_embedding = np.zeros(2048)  # self.compute_visual_embedding(keyframes)
        
        # Audio fingerprint
        audio_fp = self.compute_audio_fingerprint(url)
        
        metadata = {
            'duration': info.get('duration', 0),
            'view_count': info.get('view_count', 0),
            'upload_date': info.get('upload_date', ''),
            'channel': info.get('uploader', ''),
            'tags': info.get('tags', []),
        }
        
        return VideoFeatures(
            video_id=video_id,
            url=url,
            title=title,
            description=description,
            keyframes=keyframes,
            audio_fingerprint=audio_fp,
            visual_embedding=visual_embedding,
            text_embedding=text_embedding,
            metadata=metadata
        )


class SimilarityDetector:
    """Finds and ranks similar videos"""
    
    def __init__(self, youtube_api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        self.processor = VideoProcessor()
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search_youtube(self, query: str, max_results: int =10) -> List[Dict]:
        """Search YouTube for related videos"""
        try:
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results,
                type='video'
            ).execute()
            
            videos = []
            for item in search_response.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    videos.append({
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'channel': item['snippet']['channelTitle'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                    })
            
            return videos
        
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return []
    
    def find_similar_videos(self, source_video: VideoFeatures, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find similar videos using embeddings"""
        # Search using title and tags as query
        search_query = f"{source_video.title} {' '.join(source_video.metadata.get('tags', [])[:3])}"
        candidate_videos = self.search_youtube(search_query, max_results=20)
        
        # Compute similarity for each candidate
        similarities = []
        
        for candidate in candidate_videos:
            if candidate['video_id'] == source_video.video_id:
                continue  # Skip the source video itself
            
            # Create embedding for candidate
            candidate_text = f"{candidate['title']}. {candidate['description']}"
            candidate_embedding = self.processor.text_model.encode(candidate_text)
            
            # Compute similarity
            similarity = self.cosine_similarity(source_video.text_embedding, candidate_embedding)
            
            similarities.append((candidate, similarity))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class AIReasoningAgent:
    """Uses LLM to explain video similarities"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Anthropic API key
        For this assignment, you can use Anthropic's Claude API
        """
        self.api_key = api_key
    
    def classify_relationship(self, source: VideoFeatures, candidate: Dict, 
                             similarity: float) -> Tuple[str, str]:
        """
        Classify the relationship between videos and explain why
        Returns: (classification, reason)
        """
        
        # Rule-based classification for MVP
        if similarity > 0.95:
            classification = "re-upload"
            reason = "Extremely high similarity in content, likely identical or near-identical video"
        elif similarity > 0.85:
            classification = "edited-copy"
            reason = "Very high similarity with some differences, possibly edited version or compilation"
        elif similarity > 0.70:
            classification = "similar-content"
            reason = "Strong topical overlap, covers same subject with different approach"
        else:
            classification = "related"
            reason = "Loosely related content, shares some themes or topics"
        
        # For production, call LLM API here
        # This would analyze titles, descriptions, and provide nuanced reasoning
        
        return classification, reason
    
    def explain_similarity(self, results: List[SimilarityResult]) -> str:
        """Generate human-readable explanation of findings"""
        if not results:
            return "No similar videos found."
        
        explanation = "Found similar videos:\n\n"
        
        for i, result in enumerate(results, 1):
            explanation += f"{i}. {result.title}\n"
            explanation += f"   Similarity: {result.similarity_score:.1%}\n"
            explanation += f"   Type: {result.classification.replace('-', ' ').title()}\n"
            explanation += f"   Reason: {result.reason}\n"
            explanation += f"   URL: {result.video_url}\n\n"
        
        return explanation

class VideoSimilarityFinder:
    """Main orchestrator class"""
    
    def __init__(self, youtube_api_key: str, anthropic_api_key: str = None):
        self.detector = SimilarityDetector(youtube_api_key)
        self.agent = AIReasoningAgent(anthropic_api_key)
    
    def find_similar(self, video_url: str, top_k: int = 5) -> List[SimilarityResult]:
        """Complete pipeline to find similar videos"""
        print(f"Processing video: {video_url}")
        
        # Step 1: Process source video
        source_video = self.detector.processor.process_video(video_url)
        print(f"Extracted features for: {source_video.title}")
        
        # Step 2: Find similar videos
        similar_videos = self.detector.find_similar_videos(source_video, top_k)
        print(f"Found {len(similar_videos)} similar videos")
        
        # Step 3: Classify and explain
        results = []
        for candidate, similarity in similar_videos:
            classification, reason = self.agent.classify_relationship(
                source_video, candidate, similarity
            )
            
            result = SimilarityResult(
                video_url=candidate['url'],
                video_id=candidate['video_id'],
                title=candidate['title'],
                similarity_score=similarity,
                reason=reason,
                classification=classification
            )
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Set your API keys
    YOUTUBE_API_KEY = "YOUTUBE_API_KEY"
    ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"  # Optional
    
    # Initialize system
    finder = VideoSimilarityFinder(YOUTUBE_API_KEY, ANTHROPIC_API_KEY)
    
    # Find similar videos
    test_url = "https://www.youtube.com/shorts/ZxEpaJN8kyw"
    results = finder.find_similar(test_url, top_k=5)
    
    # Display results
    explanation = finder.agent.explain_similarity(results)
    print(explanation)
    
    # Export results as JSON
    results_json = [
        {
            'title': r.title,
            'url': r.video_url,
            'similarity': f"{r.similarity_score:.1%}",
            'classification': r.classification,
            'reason': r.reason
        }
        for r in results
    ]
    
    with open('similarity_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to similarity_results.json")