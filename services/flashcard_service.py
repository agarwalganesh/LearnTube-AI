from typing import Dict, Any
from config import Config
from models.database import db, Flashcard, Video
from services.graphs.video_analysis_graph import run_flashcards_generation

class FlashcardService:
    """Service for generating and managing revision flashcards using LangGraph & LangChain."""

    @classmethod
    def generate_flashcards_for_video(cls, video_id: int, count: int = 8) -> Dict[str, Any]:
        """
        Generate AI flashcards from video transcript and smart notes via LangGraph.
        """
        if not Config.is_ai_configured():
            return {
                'success': False,
                'flashcards': [],
                'error': 'API key is missing or invalid. Please check your AI configuration in .env.'
            }

        video = db.session.get(Video, video_id)
        if not video:
            return {'success': False, 'flashcards': [], 'error': 'Video not found.'}

        notes_summary = ""
        if video.notes and video.notes.summary:
            notes_summary = video.notes.summary

        transcript_text = video.transcript or ""
        if not transcript_text.strip():
            return {'success': False, 'flashcards': [], 'error': 'Video transcript is empty.'}

        # Run LangGraph flashcard generation
        gen_result = run_flashcards_generation(
            title=video.title,
            transcript=transcript_text,
            notes_summary=notes_summary,
            count=count
        )

        if not gen_result['success']:
            return gen_result

        cards_data = gen_result.get('flashcards', [])
        if not cards_data:
            return {'success': False, 'flashcards': [], 'error': 'No flashcards generated.'}

        try:
            Flashcard.query.filter_by(video_id=video_id).delete()

            saved_cards = []
            for card in cards_data:
                q = card.get('question', '').strip()
                a = card.get('answer', '').strip()
                if q and a:
                    flashcard_obj = Flashcard(video_id=video_id, question=q, answer=a)
                    db.session.add(flashcard_obj)
                    saved_cards.append({'question': q, 'answer': a})

            db.session.commit()

            return {
                'success': True,
                'flashcards': saved_cards,
                'error': None
            }

        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'flashcards': [],
                'error': f"Flashcard database save failed: {str(e)}"
            }
