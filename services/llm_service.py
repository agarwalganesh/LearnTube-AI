import json
import re
import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from config import Config

class LLMService:
    """Service for interacting with LLM (Groq or OpenAI) to generate notes and summaries."""
    
    @classmethod
    def get_client(cls, preferred_provider: Optional[str] = None) -> Tuple[Optional[OpenAI], str]:
        """
        Instantiate LLM client. Respects preferred_provider or Config.LLM_PROVIDER.
        Supports 'groq', 'gemini', and 'openai'.
        Returns (client, model_name).
        """
        provider = (preferred_provider or Config.LLM_PROVIDER or 'groq').lower()

        # If Gemini requested
        if provider == 'gemini' and Config.GEMINI_API_KEY and not Config.GEMINI_API_KEY.startswith('your_'):
            model = (Config.GEMINI_MODEL or 'gemini-3.6-flash').strip()
            return OpenAI(
                api_key=Config.GEMINI_API_KEY,
                base_url=Config.GEMINI_BASE_URL or 'https://generativelanguage.googleapis.com/v1beta/openai/',
                timeout=25.0
            ), model

        # If Groq requested
        if provider == 'groq' and Config.GROQ_API_KEY and not Config.GROQ_API_KEY.startswith('your_'):
            default_model = 'groq/compound-mini'
            model = (Config.GROQ_MODEL or default_model).strip() or default_model
            return OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL or 'https://api.groq.com/openai/v1'
            ), model

        # Fallbacks by available credentials
        if Config.GROQ_API_KEY and not Config.GROQ_API_KEY.startswith('your_'):
            return OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL or 'https://api.groq.com/openai/v1'
            ), (Config.GROQ_MODEL or 'groq/compound-mini').strip()

        if Config.GEMINI_API_KEY and not Config.GEMINI_API_KEY.startswith('your_'):
            return OpenAI(
                api_key=Config.GEMINI_API_KEY,
                base_url=Config.GEMINI_BASE_URL or 'https://generativelanguage.googleapis.com/v1beta/openai/',
                timeout=25.0
            ), (Config.GEMINI_MODEL or 'gemini-3.6-flash').strip()

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
            if not text or not text.strip():
                return None

            candidates = []

            # 1. Extract content from markdown code fences ```json ... ``` or ``` ... ```
            fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if fence_match:
                candidates.append(fence_match.group(1).strip())

            # 2. Extract JSON object {...}
            obj_match = re.search(r'(\{[\s\S]*\})', text)
            if obj_match:
                candidates.append(obj_match.group(1).strip())

            # 3. Stripped raw text
            cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)
            candidates.append(cleaned)

            for cand in candidates:
                if not cand:
                    continue
                for s in [cand, re.sub(r',\s*([\]\}])', r'\1', cand)]:
                    try:
                        parsed = json.loads(s, strict=False)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass

            return None

        notes_data = None
        last_error = None
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
                raw_content = response.choices[0].message.content or ""
                notes_data = _extract_json(raw_content)
                if notes_data:
                    break
            except Exception as e:
                last_error = e
                # Check for rate limit 429
                if '429' in str(e) or 'rate_limit' in str(e):
                    match = re.search(r'try again in ([\d\.]+)s', str(e))
                    wait_sec = (float(match.group(1)) + 0.5) if match else (3.0 * (attempt + 1))
                    time.sleep(wait_sec)
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

        # If Groq failed, try Gemini fallback if available
        if not notes_data and Config.GEMINI_API_KEY and not Config.GEMINI_API_KEY.startswith('your_') and 'gemini' not in model:
            try:
                gemini_client, gemini_model = cls.get_client(preferred_provider='gemini')
                if gemini_client:
                    response = gemini_client.chat.completions.create(
                        model=gemini_model,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    raw_content = response.choices[0].message.content or ""
                    notes_data = _extract_json(raw_content)
            except Exception as ge:
                print(f"[Notes] Gemini fallback attempt error: {ge}")

        if not notes_data:
            return {
                'success': False,
                'error': f'Could not parse structured notes from AI response. ({str(last_error) if last_error else ""})'
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
