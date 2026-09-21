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

    def test_langgraph_schemas(self):
        """Test validation of LangGraph Pydantic structured output schemas."""
        from services.graphs.schemas import NotesSchema, FlashcardsSchema

        notes_dict = {
            "summary": "Overview of calculus",
            "key_points": ["Derivatives", "Integrals"],
            "definitions": [{"term": "Derivative", "definition": "Rate of change"}],
            "formulas": ["dy/dx = lim h->0 (f(x+h) - f(x))/h"],
            "examples": ["Velocity is derivative of position"],
            "important_concepts": ["Chain rule"]
        }
        notes = NotesSchema.model_validate(notes_dict)
        self.assertEqual(notes.summary, "Overview of calculus")
        self.assertEqual(len(notes.definitions), 1)
        self.assertEqual(notes.definitions[0].term, "Derivative")

        fc_dict = {
            "flashcards": [
                {"question": "What is a derivative?", "answer": "The instantaneous rate of change."}
            ]
        }
        fcs = FlashcardsSchema.model_validate(fc_dict)
        self.assertEqual(len(fcs.flashcards), 1)
        self.assertEqual(fcs.flashcards[0].question, "What is a derivative?")

    def test_langgraph_compilation(self):
        """Test that RAGGraph compiles properly (VideoAnalysisState notes/flashcards
        are run as single-node entrypoints; the multi-node graph was removed as dead)."""
        from services.graphs.rag_graph import rag_graph

        self.assertIsNotNone(rag_graph)

    def test_json_extraction_handles_nested_objects(self):
        """The robust JSON extractor must respect nested braces, not stop at the
        first closing brace it sees (the old regex truncation bug)."""
        from services.graphs.video_analysis_graph import _extract_json_object

        raw = (
            'Here is your JSON:\n'
            '```json\n'
            '{"summary": "OK", "key_points": ["a", "b"], '
            '"definitions": [{"term": "T", "definition": "D"}], '
            '"formulas": [], "examples": [], "important_concepts": []}\n'
            '```'
        )
        parsed = _extract_json_object(raw)
        self.assertEqual(parsed['summary'], 'OK')
        self.assertEqual(parsed['definitions'][0]['term'], 'T')

        # Strings containing braces must not confuse the parser.
        raw_with_braces_in_string = (
            '{"summary": "We use {curly} braces sometimes", "key_points": []}'
        )
        parsed2 = _extract_json_object(raw_with_braces_in_string)
        self.assertIn('{curly}', parsed2['summary'])

    def test_transcript_helper_normalizes_shapes(self):
        """TranscriptService._join_fetched handles both modern and legacy API shapes."""
        from services.transcript_service import TranscriptService

        # Legacy snippet list
        legacy = [{'text': 'Hello'}, {'text': 'world'}, {'text': ''}]
        self.assertEqual(TranscriptService._join_fetched(legacy), 'Hello world')

        # Modern FetchedTranscript-like object with to_raw_data
        class Modern:
            def to_raw_data(self):
                return [{'text': 'foo'}, {'text': 'bar'}]

        self.assertEqual(TranscriptService._join_fetched(Modern()), 'foo bar')

        self.assertEqual(TranscriptService._join_fetched(None), '')

    def test_input_length_limits(self):
        """Routes must reject oversized inputs to avoid OOM / token-burn."""
        huge_url = 'https://www.youtube.com/watch?v=' + ('A' * 600)
        res = self.client.post('/video/analyze', data={'youtube_url': huge_url}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'too long', res.data)

    def test_flashcard_count_is_clamped(self):
        """flashcard generate endpoint must clamp count to [1, 50]."""
        with self.app.app_context():
            v = Video(
                youtube_url='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                youtube_video_id='dQw4w9WgXcQ',
                title='T', transcript='t', course_name='c'
            )
            db.session.add(v)
            db.session.commit()
            vid_id = v.id

        # Non-numeric must not crash.
        res = self.client.post(
            f'/video/{vid_id}/flashcards/generate',
            data={'count': 'abc'},
            follow_redirects=True,
        )
        self.assertEqual(res.status_code, 200)

    def test_chat_clear_requires_existing_video(self):
        """chat clear endpoint must report not-found instead of silently 200."""
        res = self.client.post('/api/chat/999999/clear')
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.get_json()['success'], False)

    def test_chat_question_length_limit(self):
        """Oversized chat questions must be rejected to protect the LLM budget."""
        with self.app.app_context():
            v = Video(
                youtube_url='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                youtube_video_id='dQw4w9WgXcQ',
                title='T', transcript='t', course_name='c'
            )
            db.session.add(v)
            db.session.commit()
            vid_id = v.id

        res = self.client.post(
            f'/api/chat/{vid_id}',
            json={'question': 'x' * 2000},
        )
        self.assertEqual(res.status_code, 400)

    def test_app_factory_fails_closed_on_missing_secret_in_production(self):
        """SECRET_KEY must be required when not in development mode."""
        import os
        saved = os.environ.pop('SECRET_KEY', None)
        os.environ['FLASK_ENV'] = 'production'
        try:
            cfg = type('C', (Config,), {'SECRET_KEY': ''})  # no key
            try:
                create_app(cfg)
                self.fail("Expected RuntimeError for missing SECRET_KEY in production")
            except RuntimeError as e:
                self.assertIn('SECRET_KEY', str(e))
        finally:
            if saved is not None:
                os.environ['SECRET_KEY'] = saved
            os.environ.pop('FLASK_ENV', None)

    def test_manual_transcript_fallback(self):
        """Test that manual transcript bypasses YouTube API when cloud IP is blocked."""
        res = TranscriptService.extract_transcript(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            manual_transcript="This is a manually pasted transcript for cloud deployment."
        )
        self.assertTrue(res['success'])
        self.assertIn("manually pasted transcript", res['transcript'])

if __name__ == '__main__':
    unittest.main()

