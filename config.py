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

    # ------------------------------------------------------------------ #
    # Cross-provider runtime fallback. When the configured primary hits a
    # 429 / 5xx, automatically try the next configured provider so a single
    # rate-limited key doesn't break the whole app.
    # ------------------------------------------------------------------ #

    @classmethod
    def get_chat_model_chain(cls, temperature: float = 0.2, max_tokens: int = None):
        """Returns an ordered list of (provider_name, ChatOpenAI) instances.

        The configured LLM_PROVIDER comes first; any other configured
        providers follow in a stable order (Groq, OpenAI, Gemini) so the
        most generous free tier (Groq) is usually reached even if the
        primary is exhausted.
        """
        from langchain_openai import ChatOpenAI

        chain = []
        seen_providers = set()

        def _groq():
            if not (cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith('your_')):
                return None
            kwargs = {
                'api_key': cls.GROQ_API_KEY,
                'base_url': cls.GROQ_BASE_URL or 'https://api.groq.com/openai/v1',
                'model': (cls.GROQ_MODEL or 'groq/compound-mini').strip(),
                'temperature': temperature,
                'max_retries': 1,
                'timeout': 30,
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        def _openai():
            if not (cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('your_')):
                return None
            kwargs = {
                'api_key': cls.OPENAI_API_KEY,
                'model': (cls.OPENAI_MODEL or 'gpt-4o-mini').strip(),
                'temperature': temperature,
                'max_retries': 1,
                'timeout': 30,
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        def _gemini():
            if not (cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith('your_')):
                return None
            kwargs = {
                'api_key': cls.GEMINI_API_KEY,
                'base_url': cls.GEMINI_BASE_URL or 'https://generativelanguage.googleapis.com/v1beta/openai/',
                'model': (cls.GEMINI_MODEL or 'gemini-3.6-flash').strip(),
                'temperature': temperature,
                'max_retries': 1,
                'timeout': 30,
            }
            if max_tokens:
                kwargs['max_tokens'] = max_tokens
            return ChatOpenAI(**kwargs)

        builders = {'groq': _groq, 'openai': _openai, 'gemini': _gemini}

        # Primary first
        primary = (cls.LLM_PROVIDER or 'groq').lower()
        if primary in builders:
            llm = builders[primary]()
            if llm is not None:
                chain.append((primary, llm))
                seen_providers.add(primary)

        # Then the rest in a stable order (Groq is the most generous free tier,
        # so we put it ahead of OpenAI/Gemini in the fallback chain).
        for name in ('groq', 'openai', 'gemini'):
            if name in seen_providers:
                continue
            llm = builders[name]()
            if llm is not None:
                chain.append((name, llm))
                seen_providers.add(name)

        return chain

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        """True if the exception looks like a rate limit / transient provider issue."""
        msg = (str(err) or '').lower()
        if not msg:
            return False
        transient_markers = (
            '429', 'rate', 'quota', 'limit', '503', '502', '504', '500',
            'unavailable', 'overloaded', 'timeout', 'timed out', 'busy',
        )
        return any(m in msg for m in transient_markers)

    @classmethod
    def invoke_with_fallback(
        cls, messages, temperature: float = 0.2, max_tokens: int = None,
        return_provider: bool = False,
    ):
        """Invoke `messages` against the fallback chain.

        Returns the AIMessage (or whatever the LLM returns) on success.
        Raises the last exception if every provider fails. Each transient
        failure moves to the next provider; non-transient failures
        (auth error, bad prompt, etc.) raise immediately.
        """
        chain = cls.get_chat_model_chain(temperature=temperature, max_tokens=max_tokens)
        if not chain:
            raise RuntimeError("No AI provider is configured. Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")

        last_error = None
        for provider_name, llm in chain:
            try:
                response = llm.invoke(messages)
                if return_provider:
                    return response, provider_name
                return response
            except Exception as e:
                last_error = e
                print(f"[LLM] {provider_name} failed: {str(e)[:300]}")
                if not cls._is_transient_error(e):
                    # Auth error, invalid key, malformed prompt — don't waste
                    # time trying other providers.
                    raise
                # else: fall through to next provider

        # All providers hit a transient failure. Surface the last one.
        assert last_error is not None
        raise last_error
