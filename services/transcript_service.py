import re
from typing import Dict, Any, List
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from .youtube_service import YouTubeService

class TranscriptService:
    """Service for extracting and cleaning transcripts from YouTube videos."""

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Clean transcript text:
        - Remove bracketed cues like [Music], [Applause], (music)
        - Collapse multiple spaces/newlines
        - Strip non-printable or noisy chars
        """
        if not text:
            return ""
            
        # Remove audio cues like [Music], [Applause], [laughter], [Inaudible]
        cleaned = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)
        # Replace multiple spaces/newlines with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def extract_transcript(cls, youtube_url: str) -> Dict[str, Any]:
        """
        Extract video transcript using LangChain's YoutubeLoader with robust fallbacks.
        
        Returns:
            dict: {
                'success': bool,
                'video_id': str,
                'title': str,
                'transcript': str,
                'raw_docs': list,
                'error': str or None
            }
        """
        is_valid, video_id, error_msg = YouTubeService.validate_url(youtube_url)
        if not is_valid:
            return {'success': False, 'error': error_msg}

        metadata = YouTubeService.get_video_metadata(video_id)
        video_title = metadata.get('title', f"Video ({video_id})")
        
        transcript_text = ""
        raw_docs = []

        # 1. First attempt: LangChain YoutubeLoader
        try:
            # Try with add_video_info=True first as specified in requirements
            try:
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=True,
                    language=["en", "en-US", "en-GB"]
                )
                raw_docs = loader.load()
            except Exception:
                # Fallback if pytube / video_info fails
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=False,
                    language=["en", "en-US", "en-GB"]
                )
                raw_docs = loader.load()

            if raw_docs and raw_docs[0].page_content:
                transcript_text = "\n".join([doc.page_content for doc in raw_docs])
                if raw_docs[0].metadata and 'title' in raw_docs[0].metadata:
                    video_title = raw_docs[0].metadata['title'] or video_title

        except Exception as e:
            # Continue to direct API fallback
            pass

        # 2. Second attempt: Direct youtube_transcript_api with broader language support
        if not transcript_text.strip():
            try:
                # Support both modern (v1.0+) and legacy (v0.x) youtube_transcript_api
                api = YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else None
                
                # Try modern instance api.fetch
                if api and hasattr(api, 'fetch'):
                    try:
                        fetched = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
                        snippets = fetched.snippets if hasattr(fetched, 'snippets') else fetched
                        transcript_text = " ".join([s.text if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                    except Exception:
                        try:
                            fetched = api.fetch(video_id)
                            snippets = fetched.snippets if hasattr(fetched, 'snippets') else fetched
                            transcript_text = " ".join([s.text if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                        except Exception:
                            pass

                # Try modern instance api.list
                if not transcript_text.strip() and api and hasattr(api, 'list'):
                    try:
                        transcript_list = api.list(video_id)
                        t_obj = None
                        try:
                            t_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                        except Exception:
                            for t in transcript_list:
                                t_obj = t
                                break
                        if t_obj:
                            fetched = t_obj.fetch()
                            snippets = fetched.snippets if hasattr(fetched, 'snippets') else fetched
                            transcript_text = " ".join([s.text if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                    except Exception:
                        pass

                # Try legacy classmethods (v0.x)
                if not transcript_text.strip():
                    if hasattr(YouTubeTranscriptApi, 'get_transcript'):
                        try:
                            items = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
                            transcript_text = " ".join([item.get('text', '') for item in items])
                        except Exception:
                            try:
                                items = YouTubeTranscriptApi.get_transcript(video_id)
                                transcript_text = " ".join([item.get('text', '') for item in items])
                            except Exception:
                                pass

                    elif hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                        try:
                            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
                            for t in t_list:
                                items = t.fetch()
                                transcript_text = " ".join([item.get('text', '') for item in items])
                                break
                        except Exception:
                            pass

            except TranscriptsDisabled:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "Subtitles/captions are disabled for this video by the creator."
                }
            except NoTranscriptFound:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "No captions or transcript found for this video. Please try a video with English subtitles."
                }
            except VideoUnavailable:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "The video is unavailable, private, or does not exist."
                }
            except CouldNotRetrieveTranscript:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "Could not retrieve the video transcript. It may be restricted or have no subtitles."
                }
            except Exception as e:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Failed to retrieve transcript: {str(e)}"
                }

        cleaned_transcript = cls.clean_text(transcript_text)
        if not cleaned_transcript:
            return {
                'success': False,
                'video_id': video_id,
                'error': "The video transcript is empty or could not be decoded."
            }

        return {
            'success': True,
            'video_id': video_id,
            'title': video_title,
            'transcript': cleaned_transcript,
            'raw_docs': raw_docs,
            'error': None
        }
