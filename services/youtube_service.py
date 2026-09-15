import re
import urllib.parse
import requests
from typing import Optional, Tuple, Dict, Any

class YouTubeService:
    """Service for validating YouTube URLs and extracting video metadata."""
    
    YOUTUBE_VIDEO_ID_REGEX = re.compile(
        r'(?:v=|\/embed\/|\/1080p\/|\/shorts\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|watch\?.+&v=)([\w-]{11})'
    )
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """Extract 11-character YouTube video ID from various URL formats."""
        if not url:
            return None
        url = url.strip()
        
        # Direct video ID passed
        if re.fullmatch(r'[\w-]{11}', url):
            return url
            
        match = cls.YOUTUBE_VIDEO_ID_REGEX.search(url)
        if match:
            return match.group(1)
            
        # Parse query params if standard parse fails
        try:
            parsed_url = urllib.parse.urlparse(url)
            if 'youtube.com' in parsed_url.hostname or 'youtu.be' in parsed_url.hostname:
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if 'v' in query_params:
                    return query_params['v'][0]
        except Exception:
            pass
            
        return None

    @classmethod
    def normalize_url(cls, video_id: str) -> str:
        """Create standard watch URL from video ID."""
        return f"https://www.youtube.com/watch?v={video_id}"

    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate URL and return (is_valid, video_id, error_message).
        """
        if not url or not url.strip():
            return False, None, "Please enter a YouTube video URL."
            
        video_id = cls.extract_video_id(url)
        if not video_id:
            return False, None, "Invalid YouTube URL. Please provide a link like https://www.youtube.com/watch?v=..."
            
        return True, video_id, None

    @classmethod
    def get_video_metadata(cls, video_id: str) -> Dict[str, Any]:
        """
        Fetch video title, author, and thumbnail via YouTube oEmbed API.
        Does not require any API key and is resilient to bot blocking.
        """
        default_meta = {
            'video_id': video_id,
            'title': f'YouTube Video ({video_id})',
            'author': 'YouTube Creator',
            'thumbnail_url': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
        }
        
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'video_id': video_id,
                    'title': data.get('title', default_meta['title']),
                    'author': data.get('author_name', default_meta['author']),
                    'thumbnail_url': data.get('thumbnail_url', default_meta['thumbnail_url'])
                }
        except Exception:
            pass
            
        return default_meta
