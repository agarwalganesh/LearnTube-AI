import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

class Config:
    """Application configuration settings."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-youtube-learning-assistant')
    
    # SQLite Database (always resolve to absolute path)
    _db_path = (BASE_DIR / "youtube_learning.db").as_posix()
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{_db_path}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # LLM Provider ('groq' or 'openai')
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').lower()
    
    # Groq (100% Free)
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    GROQ_BASE_URL = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # Embeddings ('local' for free ONNX or 'openai')
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local').lower()
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', str(BASE_DIR / 'chroma_db'))
    
    @classmethod
    def is_ai_configured(cls) -> bool:
        """Check if any valid AI provider key has been provided."""
        has_groq = bool(cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith('your_'))
        has_openai = bool(cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('your_'))
        return has_groq or has_openai

    @classmethod
    def is_openai_configured(cls) -> bool:
        """Backwards-compatible check for template and route rendering."""
        return cls.is_ai_configured()
