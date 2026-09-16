import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config
from .chroma_service import ChromaService
from models.database import db, ChatMessage, Video

class RAGService:
    """Service implementing Retrieval-Augmented Generation for grounded video Q&A."""

    @classmethod
    def get_client(cls) -> Tuple[Optional[OpenAI], str]:
        """Obtain AI client and model name (Groq or OpenAI)."""
        if Config.GROQ_API_KEY and not Config.GROQ_API_KEY.startswith('your_'):
            model = (Config.GROQ_MODEL or 'groq/compound').strip() or 'groq/compound'
            return OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL or 'https://api.groq.com/openai/v1'
            ), model

        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith('your_'):
            model = (Config.OPENAI_MODEL or 'gpt-4o-mini').strip() or 'gpt-4o-mini'
            return OpenAI(api_key=Config.OPENAI_API_KEY), model

        return None, ""

    @classmethod
    def answer_question(
        cls,
        video_id: int,
        question: str,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Execute RAG workflow for a student's question about a specific video.
        """
        if not question or not question.strip():
            return {'success': False, 'answer': 'Please enter a question.'}

        if not Config.is_ai_configured():
            return {
                'success': False,
                'answer': 'AI API key is not configured. Please set GROQ_API_KEY or OPENAI_API_KEY in .env.'
            }

        client, model = cls.get_client()
        if not client:
            return {'success': False, 'answer': 'AI client initialization failed.'}

        video = db.session.get(Video, video_id)
        if not video:
            return {'success': False, 'answer': 'Video not found in database.'}

        # 1. Similarity search in ChromaDB
        retrieved_chunks = ChromaService.similarity_search(query=question, video_id=video_id, k=k)
        
        # Fallback if vector search returned nothing
        if not retrieved_chunks and video.transcript:
            context_text = video.transcript[:3000]
        else:
            context_pieces = [f"[Excerpt {i+1}]:\n{c['content']}" for i, c in enumerate(retrieved_chunks)]
            context_text = "\n\n".join(context_pieces)

        # 2. Fetch recent chat history
        recent_messages = (
            ChatMessage.query
            .filter_by(video_id=video_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(6)
            .all()
        )
        recent_messages.reverse()

        system_instruction = (
            f"You are a dedicated AI study tutor for the video titled: '{video.title}'.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Base your answer STRICTLY on the retrieved video transcript excerpts provided in the context below.\n"
            "2. If the answer or concept is NOT mentioned or cannot be inferred from the provided context, "
            "clearly and politely state: 'This information is not covered in this video.'\n"
            "3. Do NOT hallucinate external facts or invent answers.\n"
            "4. Format your answer clearly using Markdown with bullet points, bold key terms, or code formatting where appropriate.\n"
            "5. Keep the tone helpful, encouraging, and student-friendly."
        )

        messages = [{"role": "system", "content": system_instruction}]

        for msg in recent_messages:
            messages.append({"role": msg.role, "content": msg.message})

        user_content = (
            f"--- VIDEO CONTEXT EXCERPTS ---\n{context_text}\n--- END CONTEXT ---\n\n"
            f"Student Question: {question}"
        )
        messages.append({"role": "user", "content": user_content})

        answer = ""
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
                if answer:
                    break
            except Exception as e:
                last_error = e
                if '429' in str(e) or 'rate_limit' in str(e):
                    time.sleep(2)
                    continue
                time.sleep(1)

        if not answer:
            return {
                'success': False,
                'answer': f"AI processing error: {str(last_error) if last_error else 'No response from model'}"
            }

        try:
            user_msg = ChatMessage(video_id=video_id, role='user', message=question)
            ai_msg = ChatMessage(video_id=video_id, role='assistant', message=answer)
            db.session.add(user_msg)
            db.session.add(ai_msg)
            db.session.commit()
        except Exception:
            db.session.rollback()

        return {
            'success': True,
            'answer': answer,
            'sources': [c['content'][:150] + '...' for c in retrieved_chunks]
        }
