import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config
from .schemas import NotesSchema, FlashcardsSchema

class VideoAnalysisState(TypedDict, total=False):
    title: str
    transcript: str
    count: int
    notes: Optional[Dict[str, Any]]
    flashcards: Optional[List[Dict[str, str]]]
    error: Optional[str]
    status: str

def _trim_transcript(transcript: str, max_chars: int = 7000) -> str:
    """Smartly sample transcript if too long to prevent token/TPM limits."""
    if not transcript:
        return ""
    if len(transcript) <= max_chars:
        return transcript

    part_len = max_chars // 3
    part1 = transcript[:part_len]
    mid = len(transcript) // 2
    part2 = transcript[mid - (part_len // 2) : mid + (part_len // 2)]
    part3 = transcript[-part_len:]
    return f"{part1}\n\n[... transcript excerpt ...]\n\n{part2}\n\n[... transcript excerpt ...]\n\n{part3}"


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract the first balanced JSON object from a possibly-wrapped LLM response.

    Replaces the previous `re.search(r'(\\{[\\s\\S]*\\})', ...)` which stops at the
    first `}` it sees and truncates nested objects. This implementation walks the
    string and respects quoted strings and braces.
    """
    if not raw_text:
        raise ValueError("Empty LLM response")

    text = raw_text.strip()

    # Strip optional ```json ... ``` fences.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()

    # Find the first '{' and walk to find its matching '}'.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    return json.loads(candidate, strict=False)
    raise ValueError("Unbalanced JSON braces in LLM response")

def generate_notes_node(state: VideoAnalysisState) -> Dict[str, Any]:
    """LangGraph node: Extract structured study notes using LangChain."""
    title = state.get('title', 'Educational Video')
    transcript = state.get('transcript', '')

    if not transcript or not transcript.strip():
        return {'error': 'Transcript is empty. Cannot generate notes.', 'status': 'failed'}

    trimmed = _trim_transcript(transcript)
    llm = Config.get_chat_model(temperature=0.2)
    if not llm:
        return {'error': 'AI provider not configured. Please check your API keys.', 'status': 'failed'}

    # 1. Try native with_structured_output
    try:
        structured_llm = llm.with_structured_output(NotesSchema)
        prompt = (
            f"Analyze the following video transcript and generate comprehensive academic study notes.\n\n"
            f"Video Title: {title}\n\n"
            f"Transcript:\n{trimmed}"
        )
        res: NotesSchema = structured_llm.invoke(prompt)
        if isinstance(res, NotesSchema):
            return {
                'notes': res.model_dump(),
                'status': 'notes_generated'
            }
        elif isinstance(res, dict):
            validated = NotesSchema.model_validate(res)
            return {
                'notes': validated.model_dump(),
                'status': 'notes_generated'
            }
    except Exception as se:
        print(f"[VideoAnalysisGraph] Native structured_output fallback triggered: {se}")

    # 2. Resilient fallback with JSON instruction + Pydantic validation
    system_prompt = (
        "You are an expert academic tutor. Extract structured study notes strictly from the video transcript.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "summary": "High-level comprehensive summary",\n'
        '  "key_points": ["Point 1", "Point 2"],\n'
        '  "definitions": [{"term": "Term", "definition": "Explanation"}],\n'
        '  "formulas": ["Formula 1"],\n'
        '  "examples": ["Example 1"],\n'
        '  "important_concepts": ["Concept 1"]\n'
        "}"
    )
    user_prompt = f"Video Title: {title}\n\nTranscript:\n{trimmed}"

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        raw_text = response.content if hasattr(response, 'content') else str(response)

        parsed = _extract_json_object(raw_text)
        validated = NotesSchema.model_validate(parsed)
        return {
            'notes': validated.model_dump(),
            'status': 'notes_generated'
        }
    except Exception as e:
        return {
            'error': f"Failed to generate structured notes: {str(e)}",
            'status': 'notes_failed'
        }

def generate_flashcards_node(state: VideoAnalysisState) -> Dict[str, Any]:
    """LangGraph node: Generate high-yield revision flashcards using LangChain."""
    title = state.get('title', 'Educational Video')
    transcript = state.get('transcript', '')
    count = state.get('count', 8)
    notes = state.get('notes') or {}

    notes_context = ""
    if notes:
        summary = notes.get('summary', '')
        kp = ", ".join(notes.get('key_points', [])[:5])
        notes_context = f"Notes Summary: {summary}\nKey Points: {kp}\n"

    trimmed = _trim_transcript(transcript, max_chars=5000)
    llm = Config.get_chat_model(temperature=0.3)
    if not llm:
        return {'error': 'AI provider not configured.', 'status': 'failed'}

    # 1. Try native with_structured_output
    try:
        structured_llm = llm.with_structured_output(FlashcardsSchema)
        prompt = (
            f"Generate exactly {count} high-yield revision flashcards based on this video educational material.\n\n"
            f"Video Title: {title}\n\n"
            f"{notes_context}\n"
            f"Transcript Excerpt:\n{trimmed}"
        )
        res: FlashcardsSchema = structured_llm.invoke(prompt)
        if isinstance(res, FlashcardsSchema):
            cards = [c.model_dump() for c in res.flashcards]
            return {'flashcards': cards, 'status': 'completed'}
        elif isinstance(res, dict) and 'flashcards' in res:
            validated = FlashcardsSchema.model_validate(res)
            cards = [c.model_dump() for c in validated.flashcards]
            return {'flashcards': cards, 'status': 'completed'}
    except Exception as se:
        print(f"[VideoAnalysisGraph] Flashcard structured output fallback triggered: {se}")

    # 2. Resilient fallback with JSON instruction + Pydantic validation
    system_prompt = (
        "You are an expert exam preparation educator. Create concise study flashcards based on the material.\n"
        "Return ONLY a valid JSON object matching:\n"
        '{\n  "flashcards": [\n    {"question": "Direct question", "answer": "Concise answer"}\n  ]\n}'
    )
    user_prompt = f"Video Title: {title}\n\n{notes_context}\nTranscript:\n{trimmed}\n\nCreate {count} flashcards."

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        raw_text = response.content if hasattr(response, 'content') else str(response)

        parsed = _extract_json_object(raw_text)
        validated = FlashcardsSchema.model_validate(parsed)
        cards = [c.model_dump() for c in validated.flashcards]
        return {'flashcards': cards, 'status': 'completed'}
    except Exception as e:
        return {'flashcards': [], 'error': f"Failed to generate flashcards: {str(e)}", 'status': 'flashcards_failed'}

def run_notes_generation(title: str, transcript: str) -> Dict[str, Any]:
    """Run single-node notes generation."""
    state = generate_notes_node({'title': title, 'transcript': transcript})
    if state.get('notes'):
        return {'success': True, 'data': state['notes'], 'error': None}
    return {'success': False, 'data': None, 'error': state.get('error', 'Failed to generate notes.')}

def run_flashcards_generation(title: str, transcript: str, notes_summary: str = "", count: int = 8) -> Dict[str, Any]:
    """Run single-node flashcards generation."""
    state = generate_flashcards_node({
        'title': title,
        'transcript': transcript,
        'notes': {'summary': notes_summary} if notes_summary else None,
        'count': count
    })
    cards = state.get('flashcards', [])
    if cards:
        return {'success': True, 'flashcards': cards, 'error': None}
    return {'success': False, 'flashcards': [], 'error': state.get('error', 'Failed to generate flashcards.')}
