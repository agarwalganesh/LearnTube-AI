import os
from flask import Flask, render_template
from config import Config
from models.database import db
from routes import video_bp, notes_bp, chatbot_bp, flashcard_bp, search_bp

def create_app(config_class=Config):
    """Application factory for the YouTube Learning Assistant."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize SQLAlchemy
    db.init_app(app)

    # Register Blueprints
    app.register_blueprint(video_bp)
    app.register_blueprint(notes_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(flashcard_bp)
    app.register_blueprint(search_bp)

    # Context processors for templates
    @app.context_processor
    def inject_globals():
        return {
            'openai_configured': Config.is_openai_configured()
        }

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('base.html', error_title="404 - Page Not Found", error_msg="The requested page could not be found."), 404

    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        return render_template('base.html', error_title="500 - Server Error", error_msg="An unexpected error occurred. Please try again."), 500

    # Ensure database tables exist
    with app.app_context():
        db.create_all()

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" YouTube Learning Assistant starting on http://localhost:{port}")
    print(f" OpenAI Configured: {Config.is_openai_configured()}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
