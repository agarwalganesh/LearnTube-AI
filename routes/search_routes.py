from flask import Blueprint, render_template, request, jsonify
from config import Config
from services.chroma_service import ChromaService
from models.database import Video

search_bp = Blueprint('search', __name__)

@search_bp.route('/search')
def search_page():
    """Semantic Smart Search page across all indexed video transcripts."""
    query = request.args.get('q', '').strip()
    video_id_filter = request.args.get('video_id', type=int)
    
    results = []
    if query:
        results = ChromaService.similarity_search(query=query, video_id=video_id_filter, k=6)

    # List of all videos for dropdown filter
    all_videos = Video.query.order_by(Video.title.asc()).all()

    return render_template(
        'search.html',
        query=query,
        results=results,
        all_videos=all_videos,
        selected_video_id=video_id_filter,
        openai_configured=Config.is_openai_configured()
    )

@search_bp.route('/api/search', methods=['POST'])
def search_api():
    """JSON API for real-time semantic search."""
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    video_id = data.get('video_id')
    
    if not query:
        return jsonify({'results': [], 'error': 'Empty search query'}), 400

    if not Config.is_openai_configured():
        return jsonify({
            'results': [],
            'error': 'OpenAI API key is required for semantic vector search.'
        }), 400

    results = ChromaService.similarity_search(
        query=query,
        video_id=int(video_id) if video_id else None,
        k=6
    )
    return jsonify({'results': results, 'count': len(results)})
