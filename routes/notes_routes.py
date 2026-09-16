from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from models.database import db, Video, Notes
from services.llm_service import LLMService

notes_bp = Blueprint('notes', __name__)

@notes_bp.route('/video/<int:video_id>/notes')
def view_notes(video_id):
    """Render smart notes page for the video."""
    video = db.session.get(Video, video_id)
    if not video:
        flash(f'Video #{video_id} was not found. Please select a video from your library.', 'warning')
        return redirect(url_for('video.index'))

    notes = video.notes
    return render_template(
        'notes.html',
        video=video,
        notes=notes,
        openai_configured=Config.is_ai_configured()
    )

@notes_bp.route('/video/<int:video_id>/notes/generate', methods=['POST'])
def generate_notes(video_id):
    """Generate or regenerate Smart Notes via LLM."""
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
        return redirect(url_for('notes.view_notes', video_id=video_id))

    res = LLMService.generate_smart_notes(video.title, video.transcript)
    if not res['success']:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'success': False, 'error': res['error']}), 500
        flash(res['error'], 'danger')
        return redirect(url_for('notes.view_notes', video_id=video_id))

    data = res['data']
    notes = video.notes
    if not notes:
        notes = Notes(video_id=video_id)
        db.session.add(notes)

    notes.summary = data.get('summary', '')
    notes.key_points = data.get('key_points', [])
    notes.definitions = data.get('definitions', [])
    notes.formulas = data.get('formulas', [])
    notes.examples = data.get('examples', [])
    notes.important_concepts = data.get('important_concepts', [])
    
    db.session.commit()

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': True, 'notes': notes.to_dict()})

    flash('Smart Notes successfully generated!', 'success')
    return redirect(url_for('notes.view_notes', video_id=video_id))
