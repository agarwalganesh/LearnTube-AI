from flask import Blueprint, render_template, request, jsonify
from config import Config
from services.chroma_service import ChromaService
from models.database import Video

search_bp = Blueprint('search', __name__)


def _valid_video_ids(video_ids):
    """Return the subset of `video_ids` that actually exist in SQLite.

    Used to filter out orphaned vector chunks. Replaces the previous
    `Video.query.all()` per-request call that loaded every row to build a Python set.
    """
    ids = {int(v) for v in video_ids if v is not None}
    if not ids:
        return set()
    rows = Video.query.with_entities(Video.id).filter(Video.id.in_(ids)).all()
    return {row[0] for row in rows}

def _fallback_search(query: str, video_id_filter: int = None, limit: int = 6):
    """Fallback transcript search when ChromaDB is empty or bypassed on Vercel."""
    results = []
    q_videos = Video.query
    if video_id_filter:
        q_videos = q_videos.filter_by(id=video_id_filter)
    videos = q_videos.all()
    
    query_lower = query.lower()
    for v in videos:
        if not v.transcript:
            continue
        t_lower = v.transcript.lower()
        pos = t_lower.find(query_lower)
        if pos != -1:
            start = max(0, pos - 80)
            end = min(len(v.transcript), pos + 250)
            snippet = v.transcript[start:end].strip()
            results.append({
                'content': snippet,
                'video_id': v.id,
                'youtube_video_id': v.youtube_video_id,
                'video_title': v.title,
                'course_name': v.course_name,
                'chunk_index': 0,
                'source': f"https://www.youtube.com/watch?v={v.youtube_video_id}",
                'similarity': 0.90
            })
            if len(results) >= limit:
                break
    return results

@search_bp.route('/search')
def search_page():
    """Semantic Smart Search page across all indexed video transcripts."""
    query = request.args.get('q', '').strip()
    video_id_filter = request.args.get('video_id', type=int)
    
    results = []
    if query:
        try:
            results = ChromaService.similarity_search(query=query, video_id=video_id_filter, k=6)
        except Exception:
            results = []
            
        if not results:
            results = _fallback_search(query=query, video_id_filter=video_id_filter, limit=6)

    # Filter out results whose video_id is not in SQLite to guarantee valid links
    valid_ids = _valid_video_ids(r.get('video_id') for r in results)
    results = [r for r in results if r.get('video_id') in valid_ids]

    # List of all videos for dropdown filter
    all_videos = Video.query.order_by(Video.title.asc()).all()

    return render_template(
        'search.html',
        query=query,
        results=results,
        all_videos=all_videos,
        selected_video_id=video_id_filter,
        openai_configured=Config.is_ai_configured()
    )

@search_bp.route('/api/search', methods=['POST'])
def search_api():
    """JSON API for real-time semantic search."""
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    video_id = data.get('video_id')
    
    if not query:
        return jsonify({'results': [], 'error': 'Empty search query'}), 400

    results = []
    try:
        results = ChromaService.similarity_search(
            query=query,
            video_id=int(video_id) if video_id else None,
            k=6
        )
    except Exception:
        results = []

    if not results:
        results = _fallback_search(
            query=query,
            video_id_filter=int(video_id) if video_id else None,
            limit=6
        )

    valid_ids = _valid_video_ids(r.get('video_id') for r in results)
    results = [r for r in results if r.get('video_id') in valid_ids]

    return jsonify({'results': results, 'count': len(results)})
