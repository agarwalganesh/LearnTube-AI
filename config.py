import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

class Config:
    """Application configuration settings."""
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.getenv('SECRET_KEY') or 'dev-secret-key-youtube-learning-assistant-2026-safe'
    
    # Check if running in Vercel / AWS Lambda serverless environment
    IS_VERCEL = bool(os.getenv('VERCEL') or os.getenv('AWS_LAMBDA_FUNCTION_NAME'))

    # SQLite Database (write to /tmp on serverless environments)
    if IS_VERCEL:
        SQLALCHEMY_DATABASE_URI = "sqlite:////tmp/youtube_learning.db"
        CHROMA_PERSIST_DIR = "/tmp/chroma_db"
        os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
        os.environ['HF_HOME'] = '/tmp/cache/huggingface'
        os.environ['CHROMA_CACHE_DIR'] = '/tmp/cache/chroma'
        os.environ['ONNX_CACHE_DIR'] = '/tmp/cache/onnx'
        os.environ['NUMBA_CACHE_DIR'] = '/tmp/cache/numba'
    else:
        _db_path = (BASE_DIR / "youtube_learning.db").as_posix()
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{_db_path}"
        CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', str(BASE_DIR / 'chroma_db'))

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Reject oversized request bodies early so a paste-bomb transcript or chat
    # question can't tie up a worker or burn through LLM context.
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB

    # LLM Provider ('groq', 'openai', or 'gemini')
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').lower()
    
    # Groq (100% Free)
    GROQ_API_KEY = (os.getenv('GROQ_API_KEY') or '').strip()
    GROQ_BASE_URL = (os.getenv('GROQ_BASE_URL') or 'https://api.groq.com/openai/v1').strip()
    GROQ_MODEL = (os.getenv('GROQ_MODEL') or 'groq/compound-mini').strip()
    
    # OpenAI
    OPENAI_API_KEY = (os.getenv('OPENAI_API_KEY') or '').strip()
    OPENAI_MODEL = (os.getenv('OPENAI_MODEL') or 'gpt-4o-mini').strip()

    # Google Gemini
    GEMINI_API_KEY = (os.getenv('GEMINI_API_KEY') or '').strip()
    GEMINI_BASE_URL = (os.getenv('GEMINI_BASE_URL') or 'https://generativelanguage.googleapis.com/v1beta/openai/').strip()
    GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-3.6-flash').strip()
    
    # Embeddings ('local' for free ONNX or 'openai')
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local').lower()
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # Optional YouTube Proxy for Vercel/Cloud deployments
    YOUTUBE_PROXY = os.getenv('YOUTUBE_PROXY', '')
    
    @classmethod
    def is_ai_configured(cls) -> bool:
        """Check if any valid AI provider key has been provided."""
        has_groq = bool(cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith('your_'))
        has_openai = bool(cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('your_'))
        has_gemini = bool(cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith('your_'))
        return has_groq or has_openai or has_gemini

    @classmethod
    def is_openai_configured(cls) -> bool:
        """Backwards-compatible check for template and route rendering."""
        return cls.is_ai_configured()

    @classmethod
    def get_chat_model(cls, temperature: float = 0.2, max_tokens: int = None, preferred_provider: str = None):
        """
        Returns a standardized LangChain ChatOpenAI instance configured for
        Groq, OpenAI, or Gemini.
        """
        from langchain_openai import ChatOpenAI

        provider = (preferred_provider or cls.LLM_PROVIDER or 'groq').lower()

        # 1. Preferred or primary: Groq
        if provider == 'groq' and cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith('your_'):
            model = (cls.GROQ_MODEL or 'groq/compound-mini').strip()
            kwargs = {
                'api_key': cls.GROQ_API_KEY,
                'base_url': cls.GROQ_BASE_URL or 'https://api.groq.com/openai/v1',
                'model': model,
                'temperature': temperature,
                'max_retries': 2,
                'timeout': 30
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        # 2. Preferred or fallback: OpenAI
        if provider == 'openai' and cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('your_'):
            model = (cls.OPENAI_MODEL or 'gpt-4o-mini').strip()
            kwargs = {
                'api_key': cls.OPENAI_API_KEY,
                'model': model,
                'temperature': temperature,
                'max_retries': 2,
                'timeout': 30
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        # 3. Preferred or fallback: Gemini
        if provider == 'gemini' and cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith('your_'):
            model = (cls.GEMINI_MODEL or 'gemini-3.6-flash').strip()
            kwargs = {
                'api_key': cls.GEMINI_API_KEY,
                'base_url': cls.GEMINI_BASE_URL or 'https://generativelanguage.googleapis.com/v1beta/openai/',
                'model': model,
                'temperature': temperature,
                'max_retries': 2,
                'timeout': 30
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        # Fallback to whichever key exists
        if cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith('your_'):
            model = (cls.GROQ_MODEL or 'groq/compound-mini').strip()
            return ChatOpenAI(
                api_key=cls.GROQ_API_KEY,
                base_url=cls.GROQ_BASE_URL or 'https://api.groq.com/openai/v1',
                model=model,
                temperature=temperature,
                max_retries=2,
                timeout=30
            )

        if cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('your_'):
            return ChatOpenAI(
                api_key=cls.OPENAI_API_KEY,
                model=cls.OPENAI_MODEL or 'gpt-4o-mini',
                temperature=temperature,
                max_retries=2,
                timeout=30
            )

        return None
