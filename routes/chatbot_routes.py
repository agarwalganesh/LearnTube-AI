# pyrefly: ignore [missing-import]
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from config import Config
from models.database import db, Video, ChatMessage
from services.rag_service import RAGService

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/video/<int:video_id>/chat')
def chat_view(video_id):
    """Render the AI Video Chatbot page."""
    video = db.session.get(Video, video_id)
    if not video:
        flash(f'Video #{video_id} was not found. Please select a video from your library.', 'warning')
        return redirect(url_for('video.index'))
    history = (
        ChatMessage.query
        .filter_by(video_id=video_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(50)
        .all()
    )
    history.reverse()
    return render_template(
        'chat.html',
        video=video,
        history=history,
        openai_configured=Config.is_ai_configured()
    )

@chatbot_bp.route('/api/chat/<int:video_id>', methods=['POST'])
def chat_api(video_id):
    """Handle student questions via the RAG pipeline."""
    try:
        video = db.session.get(Video, video_id)
        if not video:
            return jsonify({
                'success': False,
                'error': f'Video #{video_id} was not found in the database. Please select a valid video.'
            }), 404

        data = request.get_json(silent=True) or {}
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'success': False, 'error': 'Please enter a question.'}), 400

        # Bound question length to avoid burning tokens on accidental paste-bombs.
        if len(question) > 1000:
            return jsonify({
                'success': False,
                'error': 'Question is too long (limit 1000 characters).'
            }), 400

        if not Config.is_ai_configured():
            return jsonify({
                'success': False,
                'error': 'AI API key is missing. Please set your GROQ_API_KEY to use the chatbot.'
            }), 400

        result = RAGService.answer_question(video_id=video_id, question=question)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Server error processing question. Please try again.'
        }), 500

@chatbot_bp.route('/api/chat/<int:video_id>/clear', methods=['POST'])
def clear_chat(video_id):
    """Clear chat history for a video."""
    try:
        video = db.session.get(Video, video_id)
        if not video:
            return jsonify({'success': False, 'error': f'Video #{video_id} not found.'}), 404
        deleted = ChatMessage.query.filter_by(video_id=video_id).delete()
        db.session.commit()
        return jsonify({'success': True, 'message': f'Cleared {deleted} messages.', 'deleted_count': deleted})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Failed to clear chat history.'}), 500
