import json
import re
import time
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
                base_url=Config.GROQ_BASE_URL or 'https://api.groq.com/openai/v1'
            )
            default_model = 'groq/compound-mini' if Config.IS_VERCEL else 'groq/compound'
            model = (Config.GROQ_MODEL or default_model).strip() or default_model
            if Config.IS_VERCEL and model == 'groq/compound':
                model = 'groq/compound-mini'
            return client, model

        # 2. Fallback to OpenAI
        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith('your_'):
            model = (Config.OPENAI_MODEL or 'gpt-4o-mini').strip() or 'gpt-4o-mini'
            return OpenAI(api_key=Config.OPENAI_API_KEY), model

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

        # Smartly sample transcript if long to respect TPM rate limits (especially for Hindi/multilingual tokens)
        if len(transcript) > 8000:
            part1 = transcript[:3000]
            mid = len(transcript) // 2
            part2 = transcript[mid-1500:mid+1500]
            part3 = transcript[-3000:]
            trimmed_transcript = f"{part1}\n\n[...]\n\n{part2}\n\n[...]\n\n{part3}"
        else:
            trimmed_transcript = transcript

        system_prompt = (
            "You are an expert academic tutor. Extract structured study notes strictly from the video transcript.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "summary": "Concise overview of the video",\n'
            '  "key_points": ["Point 1", "Point 2"],\n'
            '  "definitions": [{"term": "Term", "definition": "Explanation"}],\n'
            '  "formulas": ["Formula 1"],\n'
            '  "examples": ["Example 1"],\n'
            '  "important_concepts": ["Concept 1"]\n'
            "}"
        )

        user_prompt = (
            f"Video Title: {title}\n\n"
            f"Transcript:\n{trimmed_transcript}\n\n"
            "Return the notes as a valid JSON object."
        )

        def _extract_json(text: str) -> Optional[dict]:
            if not text:
                return None
            cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)
            try:
                return json.loads(cleaned)
            except Exception:
                pass
            match = re.search(r'(\{[\s\S]*\})', cleaned)
            if match:
                candidate = match.group(1)
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
                fixed = re.sub(r',\s*([\]\}])', r'\1', candidate)
                try:
                    return json.loads(fixed)
                except Exception:
                    pass
            return None

        notes_data = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                notes_data = _extract_json(response.choices[0].message.content)
                if notes_data:
                    break
            except Exception as e:
                # If rate limited, sleep briefly and retry
                if '429' in str(e) or 'rate_limit' in str(e):
                    time.sleep(1.5)
                    continue
                # Check if Groq validator failed but generated the text in 'failed_generation'
                err_dict = getattr(e, 'body', None) or {}
                if isinstance(err_dict, dict) and 'error' in err_dict:
                    failed_gen = err_dict.get('error', {}).get('failed_generation')
                    if failed_gen:
                        notes_data = _extract_json(failed_gen)
                        if notes_data:
                            break
                time.sleep(1)

        if not notes_data:
            return {
                'success': False,
                'error': 'Could not parse structured notes from AI response. Please try again.'
            }

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
