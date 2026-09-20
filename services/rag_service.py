from typing import Dict, Any
from config import Config
from models.database import db, ChatMessage, Video
from services.graphs.rag_graph import run_rag_pipeline

class RAGService:
    """Service implementing Retrieval-Augmented Generation for grounded video Q&A via LangGraph."""

    @classmethod
    def answer_question(cls, video_id: int, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Answer user question strictly grounded in video transcript using LangGraph RAG workflow.
        Includes chat history and sources with dynamic hallucination prevention.
        """
        if not Config.is_ai_configured():
            return {
                'success': False,
                'answer': 'AI provider is not configured. Please check your API keys in .env.',
                'sources': []
            }

        video = db.session.get(Video, video_id)
        if not video:
            return {
                'success': False,
                'answer': 'Video not found in database.',
                'sources': []
            }

        # Fetch recent chat history for conversational context
        recent_messages = (
            ChatMessage.query
            .filter_by(video_id=video_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(3)
            .all()
        )
        recent_messages.reverse()
        history_list = [{'role': m.role, 'message': m.message} for m in recent_messages]

        # Execute LangGraph conversational RAG graph
        result = run_rag_pipeline(
            video_id=video_id,
            video_title=video.title,
            question=question,
            transcript=video.transcript or "",
            chat_history=history_list
        )

        answer = result.get('answer', '')
        sources = result.get('sources', [])

        if answer and result.get('success'):
            try:
                user_msg = ChatMessage(video_id=video_id, role='user', message=question)
                ai_msg = ChatMessage(video_id=video_id, role='assistant', message=answer)
                db.session.add(user_msg)
                db.session.add(ai_msg)
                db.session.commit()
            except Exception:
                db.session.rollback()

        return {
            'success': result.get('success', False),
            'answer': answer,
            'sources': sources,
            'error': result.get('error')
        }
