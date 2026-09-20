from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from models.database import db, Video, Notes, Flashcard
from services.youtube_service import YouTubeService
from services.transcript_service import TranscriptService
from services.chroma_service import ChromaService
from services.llm_service import LLMService

video_bp = Blueprint('video', __name__)

@video_bp.route('/')
def index():
    """Homepage: Input YouTube URL, select course, and see recent videos."""
    recent_videos = Video.query.order_by(Video.created_at.desc()).limit(6).all()
    courses = db.session.query(Video.course_name).distinct().all()
    courses_list = [c[0] for c in courses if c[0]]
    
    return render_template(
        'index.html',
        recent_videos=recent_videos,
        courses=courses_list,
        openai_configured=Config.is_openai_configured()
    )

@video_bp.route('/video/analyze', methods=['POST'])
def analyze_video():
    """Process a YouTube URL, extract transcript, index into ChromaDB, and save to SQLite."""
    try:
        youtube_url = request.form.get('youtube_url', '').strip()
        course_name = request.form.get('course_name', '').strip() or 'General'
        manual_transcript = request.form.get('manual_transcript', '').strip()

        if not youtube_url:
            flash('Please enter a valid YouTube video URL.', 'danger')
            return redirect(url_for('video.index'))

        # Validate URL
        is_valid, video_id, val_err = YouTubeService.validate_url(youtube_url)
        if not is_valid:
            flash(val_err, 'danger')
            return redirect(url_for('video.index'))

        # Check if video already exists in database
        existing_video = Video.query.filter_by(youtube_video_id=video_id).first()
        if existing_video:
            flash(f'"{existing_video.title}" is already in your learning library!', 'info')
            return redirect(url_for('video.view_video', video_id=existing_video.id))

        # Extract Transcript using LangChain YoutubeLoader or manual input
        extraction_result = TranscriptService.extract_transcript(youtube_url, manual_transcript=manual_transcript)
        if not extraction_result['success']:
            flash(extraction_result['error'], 'danger')
            return redirect(url_for('video.index'))


        # Save to SQLite Database
        normalized_url = YouTubeService.normalize_url(video_id)
        new_video = Video(
            youtube_url=normalized_url,
            youtube_video_id=video_id,
            title=extraction_result['title'],
            transcript=extraction_result['transcript'],
            course_name=course_name,
            completed=False
        )
        db.session.add(new_video)
        db.session.commit()

        # Index into ChromaDB if AI provider is configured
        if Config.is_ai_configured():
            try:
                chroma_res = ChromaService.index_video_transcript(
                    video_id=new_video.id,
                    youtube_video_id=new_video.youtube_video_id,
                    title=new_video.title,
                    course_name=new_video.course_name,
                    transcript=new_video.transcript
                )
                if chroma_res['success']:
                    count = chroma_res.get("chunk_count", 0)
                    if count > 0:
                        flash(f'Successfully analyzed and indexed {count} chunks in vector store!', 'success')
                    else:
                        flash('Successfully analyzed video and prepared transcript for instant AI study!', 'success')
                else:
                    flash(f'Video saved, but ChromaDB indexing had an issue: {chroma_res.get("error")}', 'warning')
            except Exception as ce:
                flash(f'Video saved, but vector indexing was skipped: {str(ce)}', 'warning')

            # Automatically attempt Smart Notes generation
            try:
                notes_res = LLMService.generate_smart_notes(new_video.title, new_video.transcript)
                if notes_res['success']:
                    data = notes_res['data']
                    notes_record = Notes(
                        video_id=new_video.id,
                        summary=data['summary'],
                        key_points=data['key_points'],
                        definitions=data['definitions'],
                        formulas=data['formulas'],
                        examples=data['examples'],
                        important_concepts=data['important_concepts']
                    )
                    db.session.add(notes_record)
                    db.session.commit()
                    flash('Smart Notes were automatically generated!', 'success')
            except Exception:
                pass
        else:
            flash('Video and transcript saved! Set your GROQ_API_KEY in .env to enable Smart Notes, Flashcards, and RAG Chat.', 'warning')

        return redirect(url_for('video.view_video', video_id=new_video.id))

    except Exception as e:
        db.session.rollback()
        import traceback
        traceback.print_exc()
        flash(f'Error analyzing video: {str(e)}', 'danger')
        return redirect(url_for('video.index'))

@video_bp.route('/video/<int:video_id>')
def view_video(video_id):
    """View video hub with embedded player, tabs for Notes, Flashcards, Chat, and full transcript."""
    video = db.session.get(Video, video_id)
    if not video:
        flash(f'Video #{video_id} was not found. Please select a video from your library.', 'warning')
        return redirect(url_for('video.index'))
    return render_template('video.html', video=video, openai_configured=Config.is_ai_configured())

@video_bp.route('/video/<int:video_id>/toggle-complete', methods=['POST'])
def toggle_complete(video_id):
    """Toggle video completed status."""
    video = db.get_or_404(Video, video_id)
    video.completed = not video.completed
    db.session.commit()
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': True, 'completed': video.completed})
        
    flash(f'Marked "{video.title}" as {"Completed" if video.completed else "In Progress"}.', 'success')
    return redirect(request.referrer or url_for('video.dashboard'))

@video_bp.route('/video/<int:video_id>/delete', methods=['POST'])
def delete_video(video_id):
    """Delete a video, associated notes, flashcards, chat history, and vector embeddings."""
    video = db.get_or_404(Video, video_id)
    title = video.title
    
    # Remove from ChromaDB
    ChromaService.delete_video_chunks(video_id)
    
    # Remove from SQLite (cascade deletes notes, flashcards, chat_messages)
    db.session.delete(video)
    db.session.commit()
    
    flash(f'Deleted "{title}" from your library.', 'info')
    return redirect(url_for('video.dashboard'))

@video_bp.route('/dashboard')
def dashboard():
    """Learning dashboard with statistics and organized video library."""
    course_filter = request.args.get('course')
    
    query = Video.query
    if course_filter:
        query = query.filter_by(course_name=course_filter)
        
    videos = query.order_by(Video.created_at.desc()).all()
    
    # Global metrics
    total_videos = Video.query.count()
    completed_videos = Video.query.filter_by(completed=True).count()
    total_notes = Notes.query.count()
    total_flashcards = Flashcard.query.count()
    
    progress_percent = int((completed_videos / total_videos * 100)) if total_videos > 0 else 0
    
    # Distinct courses for filter pills
    courses = db.session.query(Video.course_name).distinct().all()
    courses_list = [c[0] for c in courses if c[0]]

    return render_template(
        'dashboard.html',
        videos=videos,
        total_videos=total_videos,
        completed_videos=completed_videos,
        total_notes=total_notes,
        total_flashcards=total_flashcards,
        progress_percent=progress_percent,
        courses=courses_list,
        current_course=course_filter,
        openai_configured=Config.is_openai_configured()
    )
