from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Video(db.Model):
    """Stores metadata and transcript for processed YouTube videos."""
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    youtube_url = db.Column(db.String(500), nullable=False)
    youtube_video_id = db.Column(db.String(100), nullable=False, index=True)
    title = db.Column(db.String(500), nullable=False, default='Untitled Video')
    transcript = db.Column(db.Text, nullable=False)
    course_name = db.Column(db.String(200), nullable=False, default='General')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed = db.Column(db.Boolean, default=False, nullable=False)
    
    # Relationships
    notes = db.relationship('Notes', backref='video', uselist=False, cascade='all, delete-orphan')
    flashcards = db.relationship('Flashcard', backref='video', lazy='dynamic', cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', backref='video', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'youtube_url': self.youtube_url,
            'youtube_video_id': self.youtube_video_id,
            'title': self.title,
            'course_name': self.course_name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
            'completed': self.completed,
            'has_notes': self.notes is not None,
            'flashcard_count': self.flashcards.count(),
            'chat_message_count': self.chat_messages.count(),
            'transcript_snippet': (self.transcript[:200] + '...') if self.transcript else ''
        }


class Notes(db.Model):
    """Stores AI-generated structured notes for a video."""
    __tablename__ = 'notes'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False, unique=True)
    summary = db.Column(db.Text, nullable=False, default='')
    key_points = db.Column(db.JSON, nullable=False, default=list)
    definitions = db.Column(db.JSON, nullable=False, default=list)
    formulas = db.Column(db.JSON, nullable=False, default=list)
    examples = db.Column(db.JSON, nullable=False, default=list)
    important_concepts = db.Column(db.JSON, nullable=False, default=list)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'summary': self.summary,
            'key_points': self.key_points or [],
            'definitions': self.definitions or [],
            'formulas': self.formulas or [],
            'examples': self.examples or [],
            'important_concepts': self.important_concepts or [],
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        }


class Flashcard(db.Model):
    """Stores AI-generated flashcards for revision."""
    __tablename__ = 'flashcards'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False, index=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'question': self.question,
            'answer': self.answer,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        }


class ChatMessage(db.Model):
    """Stores user and assistant conversation history per video."""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'role': self.role,
            'message': self.message,
            'created_at': self.created_at.strftime('%H:%M') if self.created_at else ''
        }
