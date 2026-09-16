from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from models.database import db, Video, Flashcard
from services.flashcard_service import FlashcardService

flashcard_bp = Blueprint('flashcards', __name__)

@flashcard_bp.route('/video/<int:video_id>/flashcards')
def view_flashcards(video_id):
    """Render the revision flashcards page."""
    video = db.session.get(Video, video_id)
    if not video:
        flash(f'Video #{video_id} was not found. Please select a video from your library.', 'warning')
        return redirect(url_for('video.index'))

    cards = Flashcard.query.filter_by(video_id=video_id).order_by(Flashcard.id.asc()).all()
    cards_data = [c.to_dict() for c in cards]
    
    return render_template(
        'flashcards.html',
        video=video,
        flashcards=cards_data,
        openai_configured=Config.is_ai_configured()
    )

@flashcard_bp.route('/video/<int:video_id>/flashcards/generate', methods=['POST'])
def generate_flashcards(video_id):
    """Generate or regenerate flashcards using LLM."""
    video = db.session.get(Video, video_id)
    if not video:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'success': False, 'error': f'Video #{video_id} was not found.'}), 404
        flash(f'Video #{video_id} was not found.', 'warning')
        return redirect(url_for('video.index'))
    
    if not Config.is_ai_configured():
        msg = 'AI API key is missing. Please set GROQ_API_KEY in your environment.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'success': False, 'error': msg}), 400
        flash(msg, 'danger')
        return redirect(url_for('flashcards.view_flashcards', video_id=video_id))

    count = int(request.form.get('count', 8))
    result = FlashcardService.generate_flashcards_for_video(video_id, count=count)
    
    if not result['success']:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'success': False, 'error': result['error']}), 500
        flash(result['error'], 'danger')
        return redirect(url_for('flashcards.view_flashcards', video_id=video_id))

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': True, 'flashcards': result['flashcards']})

    flash(f'Generated {len(result["flashcards"])} revision flashcards!', 'success')
    return redirect(url_for('flashcards.view_flashcards', video_id=video_id))
