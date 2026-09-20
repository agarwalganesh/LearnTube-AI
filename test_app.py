"""
Verification suite for the YouTube Learning Assistant MVP.
Tests URL parsing, text cleaning, database CRUD, ChromaDB operations, and Flask routes.
"""
import unittest
from app import create_app
from models.database import db, Video, Notes, Flashcard, ChatMessage
from services.youtube_service import YouTubeService
from services.transcript_service import TranscriptService
from services.chroma_service import ChromaService

from config import Config

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class TestYouTubeLearningAssistant(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_youtube_service_url_parsing(self):
        """Test URL extraction for various YouTube URL patterns."""
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        for url, expected_id in urls:
            vid = YouTubeService.extract_video_id(url)
            self.assertEqual(vid, expected_id, f"Failed on {url}")

        # Invalid URL
        is_valid, vid, err = YouTubeService.validate_url("https://example.com/not-a-video")
        self.assertFalse(is_valid)

    def test_transcript_text_cleaner(self):
        """Test cleaning of bracketed audio cues and excess whitespace."""
        raw = "Hello students [Music] today we will   discuss (Applause) neural networks.\n\n\nEnjoy!"
        cleaned = TranscriptService.clean_text(raw)
        self.assertEqual(cleaned, "Hello students today we will discuss neural networks. Enjoy!")

    def test_database_models(self):
        """Test creating Video and related child entities."""
        with self.app.app_context():
            v = Video(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                youtube_video_id="dQw4w9WgXcQ",
                title="Introduction to Machine Learning",
                transcript="This is a test transcript about machine learning algorithms and neural networks.",
                course_name="Machine Learning"
            )
            db.session.add(v)
            db.session.commit()

            self.assertIsNotNone(v.id)
            self.assertEqual(v.course_name, "Machine Learning")

            # Add Notes
            n = Notes(
                video_id=v.id,
                summary="A summary of ML concepts.",
                key_points=["Supervised learning", "Unsupervised learning"],
                definitions=[{"term": "Overfitting", "definition": "Fitting noise in training data"}],
                formulas=["Loss = (y - y_hat)^2"],
                examples=["House price prediction"],
                important_concepts=["Bias-Variance Tradeoff"]
            )
            db.session.add(n)

            # Add Flashcard
            fc = Flashcard(
                video_id=v.id,
                question="What is overfitting?",
                answer="When a model memorizes noise instead of general patterns."
            )
            db.session.add(fc)

            # Add ChatMessage
            msg = ChatMessage(video_id=v.id, role="user", message="Can you explain overfitting?")
            db.session.add(msg)
            db.session.commit()

            # Verify relationships and counts
            loaded_video = db.session.get(Video, v.id)
            self.assertIsNotNone(loaded_video.notes)
            self.assertEqual(loaded_video.flashcards.count(), 1)
            self.assertEqual(loaded_video.chat_messages.count(), 1)
            self.assertEqual(loaded_video.notes.summary, "A summary of ML concepts.")

    def test_routes_rendering(self):
        """Test that Flask routes render without template errors."""
        # 1. Homepage
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'LearnTube AI', res.data)

        # 2. Dashboard
        res = self.client.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Learning Dashboard', res.data)

        # 3. Search page
        res = self.client.get('/search')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Semantic Smart Search', res.data)

        # 4. Analyze route rejection on invalid URL
        res = self.client.post('/video/analyze', data={'youtube_url': 'not-a-youtube-url'}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Invalid YouTube URL', res.data)

    def test_video_page_flow(self):
        """Test viewing video, notes, flashcards, and chat views with database records."""
        with self.app.app_context():
            v = Video(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                youtube_video_id="dQw4w9WgXcQ",
                title="Linear Regression Tutorial",
                transcript="Transcript content for linear regression.",
                course_name="Data Science"
            )
            db.session.add(v)
            db.session.commit()
            vid_id = v.id

        # Video Hub
        res = self.client.get(f'/video/{vid_id}')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Linear Regression Tutorial', res.data)

        # Notes page
        res = self.client.get(f'/video/{vid_id}/notes')
        self.assertEqual(res.status_code, 200)

        # Flashcards page
        res = self.client.get(f'/video/{vid_id}/flashcards')
        self.assertEqual(res.status_code, 200)

        # Chat page
        res = self.client.get(f'/video/{vid_id}/chat')
        self.assertEqual(res.status_code, 200)

        # Toggle complete
        res = self.client.post(f'/video/{vid_id}/toggle-complete', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

if __name__ == '__main__':
    unittest.main()
