import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from .embedding_service import EmbeddingService

class ChromaService:
    """Service for managing document embeddings and similarity search in ChromaDB."""
    
    COLLECTION_NAME = "youtube_transcripts"
    _client = None
    _collection = None

    @classmethod
    def get_client(cls):
        """Get or initialize persistent ChromaDB client."""
        if cls._client is None:
            persist_dir = Config.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)
            cls._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return cls._client

    @classmethod
    def get_collection(cls):
        """Get or create collection for transcripts."""
        if cls._collection is None:
            client = cls.get_client()
            cls._collection = client.get_or_create_collection(
                name=cls.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return cls._collection

    @classmethod
    def index_video_transcript(
        cls,
        video_id: int,
        youtube_video_id: str,
        title: str,
        course_name: str,
        transcript: str
    ) -> Dict[str, Any]:
        """
        Split transcript into chunks, compute embeddings, and store in ChromaDB with metadata.
        """
        try:
            if Config.IS_VERCEL:
                # On serverless (Vercel), storage is ephemeral and downloading 80MB ONNX
                # models exceeds Lambda execution timeouts. Safe skip.
                return {'success': True, 'chunk_count': 0, 'error': None}

            if not transcript or not transcript.strip():
                return {'success': False, 'chunk_count': 0, 'error': 'Transcript is empty'}
                
            if not Config.is_ai_configured():
                return {
                    'success': False, 
                    'chunk_count': 0, 
                    'error': 'AI API key is not configured.'
                }

            # First remove any existing chunks for this video to prevent duplicates
            try:
                cls.delete_video_chunks(video_id)
            except Exception:
                pass

            # Chunk the transcript
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(transcript)
            
            if not chunks:
                return {'success': False, 'chunk_count': 0, 'error': 'No chunks generated'}

            # Prepare metadata and document IDs
            documents = []
            metadatas = []
            ids = []

            for idx, chunk in enumerate(chunks):
                chunk_id = f"vid_{video_id}_chunk_{idx}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "video_id": int(video_id),
                    "youtube_video_id": str(youtube_video_id),
                    "video_title": str(title),
                    "course_name": str(course_name),
                    "chunk_index": int(idx),
                    "source": f"https://www.youtube.com/watch?v={youtube_video_id}"
                })

            # Generate embeddings
            embeddings = EmbeddingService.embed_documents(documents)
            
            # Store in ChromaDB
            collection = cls.get_collection()
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            return {
                'success': True,
                'chunk_count': len(chunks),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'chunk_count': 0,
                'error': f"ChromaDB indexing error: {str(e)}"
            }

    @classmethod
    def similarity_search(
        cls,
        query: str,
        video_id: Optional[int] = None,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on indexed transcripts.
        If video_id is provided, search is filtered to that specific video.
        """
        if not query or not query.strip():
            return []

        if Config.IS_VERCEL:
            return []
            
        if not Config.is_ai_configured():
            return []

        try:
            query_embedding = EmbeddingService.embed_query(query)
            collection = cls.get_collection()
            
            where_clause = None
            if video_id is not None:
                where_clause = {"video_id": int(video_id)}

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause
            )

            formatted_results = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                docs = results['documents'][0]
                metas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
                distances = results['distances'][0] if 'distances' in results and results['distances'] else []

                for i in range(len(docs)):
                    meta = metas[i] if i < len(metas) else {}
                    dist = distances[i] if i < len(distances) else 1.0
                    # For cosine distance, similarity score = 1 - distance
                    sim_score = round(max(0.0, min(1.0, 1.0 - dist)), 3)
                    
                    formatted_results.append({
                        'content': docs[i],
                        'video_id': meta.get('video_id'),
                        'youtube_video_id': meta.get('youtube_video_id'),
                        'video_title': meta.get('video_title', 'Video'),
                        'course_name': meta.get('course_name', 'General'),
                        'chunk_index': meta.get('chunk_index', 0),
                        'source': meta.get('source', ''),
                        'similarity': sim_score
                    })

            return formatted_results

        except Exception:
            return []

    @classmethod
    def delete_video_chunks(cls, video_id: int) -> bool:
        """Remove all chunks associated with a specific video_id."""
        try:
            collection = cls.get_collection()
            collection.delete(where={"video_id": int(video_id)})
            return True
        except Exception:
            return False

    @classmethod
    def get_total_chunks(cls) -> int:
        """Get total number of chunks stored in ChromaDB."""
        try:
            collection = cls.get_collection()
            return collection.count()
        except Exception:
            return 0
