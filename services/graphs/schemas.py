from typing import List, Optional
from pydantic import BaseModel, Field

class DefinitionItem(BaseModel):
    term: str = Field(..., description="The key term or concept name")
    definition: str = Field(..., description="Clear, concise explanation of the term")

class NotesSchema(BaseModel):
    """Structured academic study notes extracted from video transcript."""
    summary: str = Field(..., description="High-level comprehensive overview and summary of the video topic")
    key_points: List[str] = Field(default_factory=list, description="List of the most important takeaways from the video")
    definitions: List[DefinitionItem] = Field(default_factory=list, description="List of key technical terms, vocabulary, and definitions")
    formulas: List[str] = Field(default_factory=list, description="Mathematical formulas, algorithms, or equations mentioned")
    examples: List[str] = Field(default_factory=list, description="Real-world examples, analogies, or case studies used by the instructor")
    important_concepts: List[str] = Field(default_factory=list, description="Critical core concepts necessary for exam mastery")

class FlashcardItem(BaseModel):
    question: str = Field(..., description="Direct, focused exam-style question testing a single concept")
    answer: str = Field(..., description="Concise, accurate answer easy to memorize for exams")

class FlashcardsSchema(BaseModel):
    """Collection of revision flashcards for active recall study."""
    flashcards: List[FlashcardItem] = Field(..., description="List of revision flashcard Q&A pairs")
