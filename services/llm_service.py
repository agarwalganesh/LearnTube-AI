import json
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config

class LLMService:
    """Service for interacting with LLM (Groq or OpenAI) to generate notes and summaries."""
    
    @classmethod
    def get_client(cls) -> Tuple[Optional[OpenAI], str]:
        """
        Instantiate LLM client. Prefers Groq if configured, otherwise OpenAI.
        Returns (client, model_name).
        """
        # 1. Try Groq (100% Free)
        if Config.GROQ_API_KEY and not Config.GROQ_API_KEY.startswith('your_'):
            client = OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL
            )
            return client, Config.GROQ_MODEL

        # 2. Fallback to OpenAI
        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith('your_'):
            return OpenAI(api_key=Config.OPENAI_API_KEY), Config.OPENAI_MODEL

        return None, ""

    @classmethod
    def generate_smart_notes(cls, title: str, transcript: str) -> Dict[str, Any]:
        """
        Generate structured study notes from video transcript.
        """
        if not Config.is_ai_configured():
            return {
                'success': False,
                'error': 'API key is missing. Please set GROQ_API_KEY or OPENAI_API_KEY in .env.'
            }

        if not transcript or not transcript.strip():
            return {
                'success': False,
                'error': 'Transcript is empty. Cannot generate notes.'
            }

        client, model = cls.get_client()
        if not client:
            return {'success': False, 'error': 'Failed to initialize AI client.'}

        # Smartly sample transcript if excessively long to respect TPM rate limits
        if len(transcript) > 20000:
            part1 = transcript[:7000]
            mid = len(transcript) // 2
            part2 = transcript[mid-3500:mid+3500]
            part3 = transcript[-7000:]
            trimmed_transcript = f"{part1}\n\n[...]\n\n{part2}\n\n[...]\n\n{part3}"
        else:
            trimmed_transcript = transcript

        system_prompt = (
            "You are an expert academic tutor and study assistant. Your job is to extract comprehensive, "
            "clear, and structured study notes strictly from the provided YouTube video transcript.\n\n"
            "STRICT RULES:\n"
            "1. Grounding: Rely ONLY on the information present in the video transcript. DO NOT hallucinate, assume, "
            "or invent details that are not in the video.\n"
            "2. If a section (like formulas or specific examples) is not discussed or present in the video, "
            "return an empty list [] for that key.\n"
            "3. Format your entire output as a valid JSON object matching this schema exactly:\n"
            "{\n"
            '  "summary": "Concise overview of the complete video content",\n'
            '  "key_points": ["Point 1", "Point 2", ...],\n'
            '  "definitions": [{"term": "Term Name", "definition": "Explanation as taught in video"}, ...],\n'
            '  "formulas": ["Formula 1", ...],\n'
            '  "examples": ["Example 1 explained by instructor", ...],\n'
            '  "important_concepts": ["Concept 1", ...]\n'
            "}"
        )

        user_prompt = (
            f"Video Title: {title}\n\n"
            f"Transcript Content:\n{trimmed_transcript}\n\n"
            "Generate the structured study notes in strict JSON format."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = response.choices[0].message.content
            notes_data = json.loads(content)

            normalized_notes = {
                "summary": str(notes_data.get("summary", "")),
                "key_points": list(notes_data.get("key_points", [])),
                "definitions": list(notes_data.get("definitions", [])),
                "formulas": list(notes_data.get("formulas", [])),
                "examples": list(notes_data.get("examples", [])),
                "important_concepts": list(notes_data.get("important_concepts", []))
            }

            return {
                'success': True,
                'data': normalized_notes,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"AI API error during note generation: {str(e)}"
            }
