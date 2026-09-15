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

        # 2. Second attempt: Direct youtube_transcript_api with broader language support & auto-translation
        if not transcript_text.strip():
            last_error = None
            try:
                api = YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else None
                
                # Fetch available transcript list
                if api and hasattr(api, 'list'):
                    try:
                        transcript_list = api.list(video_id)
                        t_obj = None
                        
                        # Priority 1: English variants (manual or auto-generated)
                        try:
                            t_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU'])
                        except Exception:
                            pass

                        # Priority 2: Auto-translate any available transcript to English
                        if not t_obj:
                            for t in transcript_list:
                                if getattr(t, 'is_translatable', False):
                                    try:
                                        t_obj = t.translate('en')
                                        break
                                    except Exception:
                                        pass

                        # Priority 3: Any available transcript in its original language
                        if not t_obj:
                            for t in transcript_list:
                                t_obj = t
                                break

                        if t_obj:
                            fetched = t_obj.fetch()
                            snippets = getattr(fetched, 'snippets', fetched)
                            if hasattr(fetched, 'to_raw_data'):
                                raw_snippets = fetched.to_raw_data()
                                transcript_text = " ".join([s.get('text', '') for s in raw_snippets if s.get('text')])
                            else:
                                transcript_text = " ".join([getattr(s, 'text', '') if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                    except Exception as le:
                        last_error = str(le)

                # Fallback to direct fetch if list was not used or returned empty
                if not transcript_text.strip() and api and hasattr(api, 'fetch'):
                    for lang_option in [['en', 'en-US', 'en-GB', 'en-IN'], None]:
                        try:
                            if lang_option:
                                fetched = api.fetch(video_id, languages=lang_option)
                            else:
                                fetched = api.fetch(video_id)
                            if hasattr(fetched, 'to_raw_data'):
                                transcript_text = " ".join([s.get('text', '') for s in fetched.to_raw_data() if s.get('text')])
                            else:
                                snippets = getattr(fetched, 'snippets', fetched)
                                transcript_text = " ".join([getattr(s, 'text', '') if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                            if transcript_text.strip():
                                break
                        except Exception as fe:
                            last_error = str(fe)

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
                    'error': "No captions or transcript found for this video. Please try a video with English or auto-generated subtitles."
                }
            except VideoUnavailable:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "The video is unavailable, private, or does not exist."
                }
            except CouldNotRetrieveTranscript as cne:
                err_str = str(cne)
                if "IpBlocked" in err_str or "RequestBlocked" in err_str or "blocked" in err_str.lower():
                    return {
                        'success': False,
                        'video_id': video_id,
                        'error': "YouTube blocked requests from cloud server IP (Vercel). Run the app locally via 'python app.py' on your computer where it works 100%!"
                    }
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Could not retrieve video transcript: {err_str}"
                }
            except Exception as e:
                err_str = str(e)
                if "IpBlocked" in err_str or "RequestBlocked" in err_str or "blocked" in err_str.lower():
                    return {
                        'success': False,
                        'video_id': video_id,
                        'error': "YouTube blocked requests from cloud server IP (Vercel). Run the app locally via 'python app.py' on your computer where it works 100%!"
                    }
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Failed to retrieve transcript: {err_str}"
                }

        cleaned_transcript = cls.clean_text(transcript_text)
        if not cleaned_transcript:
            if last_error:
                if "IpBlocked" in last_error or "RequestBlocked" in last_error or "blocked" in last_error.lower():
                    return {
                        'success': False,
                        'video_id': video_id,
                        'error': "YouTube blocked requests from cloud server IP (Vercel). Run the app locally via 'python app.py' on your computer where it works 100%!"
                    }
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Could not extract transcript: {last_error}"
                }
            return {
                'success': False,
                'video_id': video_id,
                'error': "No subtitles found for this video. Please try an educational video with subtitles enabled."
            }

        return {
            'success': True,
            'video_id': video_id,
            'title': video_title,
            'transcript': cleaned_transcript,
            'raw_docs': raw_docs,
            'error': None
        }
