import json
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config
from models.database import db, Flashcard, Video

class FlashcardService:
    """Service for generating and managing revision flashcards using Groq or OpenAI."""

    @classmethod
    def get_client(cls) -> Tuple[Optional[OpenAI], str]:
        """Obtain AI client and model name (Groq or OpenAI)."""
        if Config.GROQ_API_KEY and not Config.GROQ_API_KEY.startswith('your_'):
            return OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL
            ), Config.GROQ_MODEL

        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith('your_'):
            return OpenAI(api_key=Config.OPENAI_API_KEY), Config.OPENAI_MODEL

        return None, ""

    @classmethod
    def generate_flashcards_for_video(cls, video_id: int, count: int = 8) -> Dict[str, Any]:
        """
        Generate AI flashcards from video transcript and smart notes.
        """
        if not Config.is_ai_configured():
            return {
                'success': False,
                'flashcards': [],
                'error': 'API key is missing or invalid. Please set GROQ_API_KEY or OPENAI_API_KEY in .env.'
            }

        video = db.session.get(Video, video_id)
        if not video:
            return {'success': False, 'flashcards': [], 'error': 'Video not found.'}

        client, model = cls.get_client()
        if not client:
            return {'success': False, 'flashcards': [], 'error': 'Failed to initialize AI client.'}

        notes_summary = ""
        if video.notes:
            notes_summary = (
                f"Notes Summary: {video.notes.summary}\n"
                f"Key Points: {', '.join(video.notes.key_points or [])}\n"
                f"Formulas: {', '.join(video.notes.formulas or [])}\n"
            )

        if video.transcript:
            if len(video.transcript) > 20000:
                part1 = video.transcript[:7000]
                mid = len(video.transcript) // 2
                part2 = video.transcript[mid-3500:mid+3500]
                part3 = video.transcript[-7000:]
                transcript_excerpt = f"{part1}\n\n[...]\n\n{part2}\n\n[...]\n\n{part3}"
            else:
                transcript_excerpt = video.transcript
        else:
            transcript_excerpt = ""

        system_prompt = (
            "You are an expert exam preparation educator. Create concise, high-yield study flashcards strictly "
            "based on the provided educational video material.\n\n"
            "GUIDELINES:\n"
            "1. Each flashcard must test one clear concept, definition, formula, or instructor example.\n"
            "2. Keep the questions direct and engaging.\n"
            "3. Keep the answers concise, accurate, and easy to memorize for an exam.\n"
            "4. Only use information provided in the transcript and notes. No hallucinations.\n"
            "5. Return a JSON object with a 'flashcards' list of objects:\n"
            '{\n  "flashcards": [\n    {"question": "What is ...?", "answer": "..."}\n  ]\n}'
        )

        user_prompt = (
            f"Video Title: {video.title}\n\n"
            f"{notes_summary}\n"
            f"Transcript Excerpt:\n{transcript_excerpt}\n\n"
            f"Generate {count} high-quality flashcards in JSON format."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            cards_data = parsed.get("flashcards", [])

            if not cards_data:
                return {'success': False, 'flashcards': [], 'error': 'No flashcards were generated.'}

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
                'error': f"Flashcard generation failed: {str(e)}"
            }
