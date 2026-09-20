import os
import re
from typing import Dict, Any, List, Optional
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
try:
    from youtube_transcript_api.proxies import GenericProxyConfig
except ImportError:
    GenericProxyConfig = None

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
            
        cleaned = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def _get_api(cls) -> Any:
        """Initialize YouTubeTranscriptApi with proxy if configured in environment."""
        proxy_url = os.getenv('YOUTUBE_PROXY') or os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        if proxy_url and GenericProxyConfig:
            try:
                proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
                return YouTubeTranscriptApi(proxy_config=proxy_config)
            except Exception:
                pass
        return YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else None

    @classmethod
    def extract_transcript(cls, youtube_url: str, manual_transcript: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract video transcript using manual text, LangChain's YoutubeLoader, or direct API.
        
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

        # 1. If user provided a manual transcript fallback (useful when YouTube blocks cloud IPs)
        if manual_transcript and manual_transcript.strip():
            cleaned = cls.clean_text(manual_transcript)
            if cleaned:
                return {
                    'success': True,
                    'video_id': video_id,
                    'title': video_title,
                    'transcript': cleaned,
                    'raw_docs': [],
                    'error': None
                }

        transcript_text = ""
        raw_docs = []

        # 2. First attempt: LangChain YoutubeLoader
        try:
            try:
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=True,
                    language=["en", "en-US", "en-GB", "en-IN", "hi"]
                )
                raw_docs = loader.load()
            except Exception:
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=False,
                    language=["en", "en-US", "en-GB", "en-IN", "hi"]
                )
                raw_docs = loader.load()

            if raw_docs and raw_docs[0].page_content:
                transcript_text = "\n".join([doc.page_content for doc in raw_docs])
                if raw_docs[0].metadata and 'title' in raw_docs[0].metadata:
                    video_title = raw_docs[0].metadata['title'] or video_title

        except Exception:
            pass

        # 3. Second attempt: Direct youtube_transcript_api with native transcript priority
        if not transcript_text.strip():
            last_error = None
            try:
                api = cls._get_api()
                
                if api and hasattr(api, 'list'):
                    try:
                        transcript_list = api.list(video_id)
                        t_obj = None
                        
                        try:
                            t_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU'])
                        except Exception:
                            pass

                        if t_obj:
                            try:
                                fetched = t_obj.fetch()
                                if hasattr(fetched, 'to_raw_data'):
                                    transcript_text = " ".join([s.get('text', '') for s in fetched.to_raw_data() if s.get('text')])
                                else:
                                    snippets = getattr(fetched, 'snippets', fetched)
                                    transcript_text = " ".join([getattr(s, 'text', '') if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                            except Exception as fe:
                                last_error = str(fe)

                        if not transcript_text.strip():
                            for t in transcript_list:
                                try:
                                    fetched = t.fetch()
                                    if hasattr(fetched, 'to_raw_data'):
                                        transcript_text = " ".join([s.get('text', '') for s in fetched.to_raw_data() if s.get('text')])
                                    else:
                                        snippets = getattr(fetched, 'snippets', fetched)
                                        transcript_text = " ".join([getattr(s, 'text', '') if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                                    if transcript_text.strip():
                                        break
                                except Exception as fe:
                                    last_error = str(fe)

                        if not transcript_text.strip():
                            for t in transcript_list:
                                if getattr(t, 'is_translatable', False):
                                    try:
                                        t_trans = t.translate('en')
                                        fetched = t_trans.fetch()
                                        if hasattr(fetched, 'to_raw_data'):
                                            transcript_text = " ".join([s.get('text', '') for s in fetched.to_raw_data() if s.get('text')])
                                        else:
                                            snippets = getattr(fetched, 'snippets', fetched)
                                            transcript_text = " ".join([getattr(s, 'text', '') if hasattr(s, 'text') else str(s.get('text', '')) for s in snippets])
                                        if transcript_text.strip():
                                            break
                                    except Exception as te:
                                        last_error = str(te)

                    except Exception as le:
                        last_error = str(le)

                if not transcript_text.strip() and api and hasattr(api, 'fetch'):
                    for lang_option in [['en', 'en-US', 'en-GB', 'en-IN', 'hi'], None]:
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
                    'error': "Subtitles/captions are disabled for this video. You can paste the transcript manually below!"
                }
            except NoTranscriptFound:
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': "No captions found for this video. You can paste the transcript text manually below!"
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
                        'error': "YouTube blocked requests from cloud server IP. Please paste the transcript text below directly in the Manual Transcript box!"
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
                        'error': "YouTube blocked requests from cloud server IP. Please paste the transcript text below directly in the Manual Transcript box!"
                    }
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Failed to retrieve transcript: {err_str}"
                }

        cleaned_transcript = cls.clean_text(transcript_text)
        if not cleaned_transcript:
            if last_error:
                err_lower = last_error.lower()
                if "ipblocked" in err_lower or "requestblocked" in err_lower or "blocked" in err_lower or "bot" in err_lower:
                    return {
                        'success': False,
                        'video_id': video_id,
                        'error': "YouTube blocked automatic transcript extraction from this server IP. Please use the 'Paste Transcript Manually' box below to proceed with this video!"
                    }
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': f"Could not retrieve video transcript ({last_error}). Please paste the transcript text in the 'Manual Transcript' box below!"
                }
            return {
                'success': False,
                'video_id': video_id,
                'error': "No captions found for this video. Please paste the transcript text directly in the 'Manual Transcript' box below."
            }


        return {
            'success': True,
            'video_id': video_id,
            'title': video_title,
            'transcript': cleaned_transcript,
            'raw_docs': raw_docs,
            'error': None
        }
