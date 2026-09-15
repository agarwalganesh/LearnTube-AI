from .video_routes import video_bp
from .notes_routes import notes_bp
from .chatbot_routes import chatbot_bp
from .flashcard_routes import flashcard_bp
from .search_routes import search_bp

__all__ = [
    'video_bp',
    'notes_bp',
    'chatbot_bp',
    'flashcard_bp',
    'search_bp'
]
