from flask import Blueprint, render_template, request, jsonify
from config import Config
from models.database import db, Video, ChatMessage
from services.rag_service import RAGService

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/video/<int:video_id>/chat')
def chat_view(video_id):
    """Render the AI Video Chatbot page."""
    video = db.get_or_404(Video, video_id)
    history = ChatMessage.query.filter_by(video_id=video_id).order_by(ChatMessage.created_at.asc()).all()
    return render_template(
        'chat.html',
        video=video,
        history=history,
        openai_configured=Config.is_openai_configured()
    )

@chatbot_bp.route('/api/chat/<int:video_id>', methods=['POST'])
def chat_api(video_id):
    """Handle student questions via the RAG pipeline."""
    video = db.get_or_404(Video, video_id)
    data = request.get_json() or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'success': False, 'error': 'Please enter a question.'}), 400

    if not Config.is_openai_configured():
        return jsonify({
            'success': False,
            'error': 'OpenAI API key is missing. Please add your key to the .env file to use the chatbot.'
        }), 400

    result = RAGService.answer_question(video_id=video_id, question=question)
    return jsonify(result)

@chatbot_bp.route('/api/chat/<int:video_id>/clear', methods=['POST'])
def clear_chat(video_id):
    """Clear chat history for a video."""
    ChatMessage.query.filter_by(video_id=video_id).delete()
    db.session.commit()
    return jsonify({'success': True, 'message': 'Chat history cleared.'})
