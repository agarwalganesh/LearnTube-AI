from typing import Dict, Any
from config import Config
from services.graphs.video_analysis_graph import run_notes_generation

class LLMService:
    """Service for interacting with LLM via LangChain & LangGraph to generate structured notes."""

    @classmethod
    def generate_smart_notes(cls, title: str, transcript: str) -> Dict[str, Any]:
        """
        Generate structured academic study notes from video transcript using LangGraph.
        """
        if not Config.is_ai_configured():
            return {
                'success': False,
                'data': None,
                'error': 'API key is missing. Please set your AI provider key in .env.'
            }

        if not transcript or not transcript.strip():
            return {
                'success': False,
                'data': None,
                'error': 'Transcript is empty. Cannot generate notes.'
            }

        result = run_notes_generation(title=title, transcript=transcript)
        return result
