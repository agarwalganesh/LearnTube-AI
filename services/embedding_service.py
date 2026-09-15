from typing import List, Optional
import chromadb.utils.embedding_functions as ef
from config import Config

class EmbeddingService:
    """Service for computing vector embeddings using local ONNX or OpenAI."""
    
    _local_ef = None
    _openai_ef = None

    @classmethod
    def get_local_model(cls):
        """Get or initialize local free ONNX embedding function."""
        if cls._local_ef is None:
            cls._local_ef = ef.DefaultEmbeddingFunction()
        return cls._local_ef

    @classmethod
    def get_openai_model(cls):
        """Get or initialize OpenAI embeddings model."""
        if cls._openai_ef is None:
            from langchain_openai import OpenAIEmbeddings
            cls._openai_ef = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                api_key=Config.OPENAI_API_KEY
            )
        return cls._openai_ef

    @classmethod
    def embed_query(cls, text: str) -> List[float]:
        """Generate vector embedding for a single search query or question."""
        if Config.EMBEDDING_PROVIDER == 'local' or not Config.OPENAI_API_KEY:
            fn = cls.get_local_model()
            embs = fn([text])
            return [float(x) for x in embs[0]]
        else:
            return cls.get_openai_model().embed_query(text)

    @classmethod
    def embed_documents(cls, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for multiple text documents."""
        if Config.EMBEDDING_PROVIDER == 'local' or not Config.OPENAI_API_KEY:
            fn = cls.get_local_model()
            embs = fn(texts)
            return [[float(x) for x in vec] for vec in embs]
        else:
            return cls.get_openai_model().embed_documents(texts)
