import os
from flask import Flask, render_template, request, jsonify
from config import Config
from models.database import db
from routes import video_bp, notes_bp, chatbot_bp, flashcard_bp, search_bp

def create_app(config_class=Config):
    """Application factory for the YouTube Learning Assistant."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure secret_key is never empty for session/flash.
    # In production, fail closed rather than silently using a known default.
    secret = app.config.get('SECRET_KEY')
    if not secret:
        is_dev = app.config.get('DEBUG') or os.getenv('FLASK_ENV', '').lower() == 'development'
        if is_dev:
            secret = 'dev-secret-key-youtube-learning-assistant-2026-safe'
        else:
            raise RuntimeError(
                "SECRET_KEY is not set. Define it in your environment or .env file "
                "before running in production."
            )
    app.config['SECRET_KEY'] = secret
    app.secret_key = secret

    # Ensure database is present: copy bundled starter_db if not present or empty
    import shutil
    from pathlib import Path
    starter_db = Config.BASE_DIR / 'data' / 'starter_db.sqlite'
    if Config.IS_VERCEL:
        tmp_db = Path('/tmp/youtube_learning.db')
        if starter_db.exists() and (not tmp_db.exists() or tmp_db.stat().st_size == 0):
            try:
                shutil.copyfile(starter_db, tmp_db)
                print(f"[Vercel Init] Seeded /tmp database from {starter_db}")
            except Exception as e:
                print(f"[Vercel Init] Failed copying starter_db: {e}")
    else:
        local_db = Config.BASE_DIR / 'youtube_learning.db'
        if starter_db.exists() and (not local_db.exists() or local_db.stat().st_size == 0):
            try:
                shutil.copyfile(starter_db, local_db)
                print(f"[Local Init] Initialized database from {starter_db}")
            except Exception as e:
                print(f"[Local Init] Failed copying starter_db: {e}")

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
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'API endpoint or resource not found (404)'}), 404
        return render_template('base.html', error_title="404 - Page Not Found", error_msg="The requested page could not be found."), 404

    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Internal server error (500). Please try again.'}), 500
        return render_template('base.html', error_title="500 - Server Error", error_msg="An unexpected error occurred. Please try again."), 500

    # Ensure database tables exist
    with app.app_context():
        db.create_all()
        app._db_ready = True

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" YouTube Learning Assistant starting on http://localhost:{port}")
    print(f" OpenAI Configured: {Config.is_openai_configured()}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
