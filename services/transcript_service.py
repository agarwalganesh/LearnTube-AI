import os
import re
import requests
from typing import Dict, Any, List, Optional, Tuple
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

# Default third-party transcript provider. Used as a fallback when YouTube
# directly blocks cloud server IPs. Override via the TRANSCRIPT_PROVIDER_URL
# environment variable (e.g. point it at a self-hosted service).
DEFAULT_THIRDPARTY_URL = os.getenv('TRANSCRIPT_PROVIDER_URL', 'https://yt.lemnoslife.com')


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
    def _get_api(cls, proxy_url: Optional[str] = None) -> Any:
        """Initialize YouTubeTranscriptApi with proxy if configured.

        LangChain's YoutubeLoader does not honor HTTP proxy env vars, so when a
        proxy is configured we deliberately bypass it later in `extract_transcript`
        and route this call through `youtube-transcript-api` with the proxy config.
        """
        if proxy_url and GenericProxyConfig:
            try:
                proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
                return YouTubeTranscriptApi(proxy_config=proxy_config)
            except Exception as e:
                print(f"[TranscriptService] Proxy config failed, falling back: {e}")
        try:
            return YouTubeTranscriptApi()
        except Exception as e:
            print(f"[TranscriptService] YouTubeTranscriptApi init failed: {e}")
            return None

    @staticmethod
    def _join_fetched(fetched: Any) -> str:
        """Normalize fetched transcript object into a single concatenated string.

        The `youtube-transcript-api` library has shipped both the modern `FetchedTranscript`
        object (with `.to_raw_data()`) and the older snippet-list API in different releases.
        This helper handles both shapes and ignores blank segments.
        """
        if fetched is None:
            return ""
        snippets = None
        if hasattr(fetched, 'to_raw_data'):
            snippets = fetched.to_raw_data()
        else:
            snippets = getattr(fetched, 'snippets', fetched)

        parts = []
        for s in snippets:
            if isinstance(s, dict):
                txt = s.get('text', '')
            else:
                txt = getattr(s, 'text', '')
            if txt:
                parts.append(txt)
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Provider implementations. Each returns (transcript_text, error_str).
    # An empty transcript_text + None error means "no transcript, try next".
    # An empty transcript_text + error_str means "definitive failure".
    # ------------------------------------------------------------------ #

    @classmethod
    def _fetch_via_langchain(cls, youtube_url: str) -> Tuple[str, List[Any], Optional[str]]:
        """LangChain YoutubeLoader. Returns (text, raw_docs, error)."""
        raw_docs = []
        try:
            try:
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=True,
                    language=["en", "en-US", "en-GB", "en-IN", "hi"]
                )
                raw_docs = loader.load()
            except Exception as e:
                # Retry without video metadata fetch (which can fail on its own).
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=False,
                    language=["en", "en-US", "en-GB", "en-IN", "hi"]
                )
                raw_docs = loader.load()

            if raw_docs and raw_docs[0].page_content:
                transcript_text = "\n".join([doc.page_content for doc in raw_docs])
                return transcript_text, raw_docs, None
            return "", raw_docs, "empty_load"
        except Exception as e:
            return "", raw_docs, f"langchain({e})"

    @classmethod
    def _fetch_via_youtube_transcript_api(
        cls, video_id: str, proxy_url: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """Direct youtube-transcript-api call, optionally through a proxy."""
        try:
            api = cls._get_api(proxy_url=proxy_url)

            if api and hasattr(api, 'list'):
                try:
                    transcript_list = api.list(video_id)

                    t_obj = None
                    # find_transcript raises NoTranscriptFound when no English
                    # variant exists; we want to silently fall through to other
                    # languages in that case.
                    try:
                        t_obj = transcript_list.find_transcript(
                            ['en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU']
                        )
                    except Exception:
                        t_obj = None

                    if t_obj:
                        try:
                            text = cls._join_fetched(t_obj.fetch())
                            if text.strip():
                                return text, None
                        except Exception as e:
                            return "", f"fetch_en({e})"

                    # No English transcript available — try the first transcript in any language.
                    if not t_obj:
                        for t in transcript_list:
                            try:
                                text = cls._join_fetched(t.fetch())
                                if text.strip():
                                    return text, None
                            except Exception:
                                continue

                    for t in transcript_list:
                        if getattr(t, 'is_translatable', False):
                            try:
                                text = cls._join_fetched(t.translate('en').fetch())
                                if text.strip():
                                    return text, None
                            except Exception:
                                continue
                except Exception as e:
                    return "", f"list({e})"

            if api and hasattr(api, 'fetch'):
                for lang_option in [['en', 'en-US', 'en-GB', 'en-IN', 'hi'], None]:
                    try:
                        fetched = api.fetch(video_id, languages=lang_option) if lang_option else api.fetch(video_id)
                        text = cls._join_fetched(fetched)
                        if text.strip():
                            return text, None
                    except Exception:
                        continue

            return "", "no_transcript_found"
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            raise
        except CouldNotRetrieveTranscript as cne:
            err_str = str(cne)
            if "IpBlocked" in err_str or "RequestBlocked" in err_str or "blocked" in err_str.lower():
                return "", "ip_blocked"
            return "", f"could_not_retrieve({err_str})"
        except Exception as e:
            err_str = str(e)
            if "IpBlocked" in err_str or "RequestBlocked" in err_str or "blocked" in err_str.lower():
                return "", "ip_blocked"
            return "", f"transcript_api({err_str})"

    @classmethod
    def _fetch_via_third_party(cls, video_id: str) -> Tuple[str, Optional[str]]:
        """Fallback to a third-party transcript provider.

        The default provider (yt.lemnoslife.com) is a public YouTube metadata
        service that some users have reported works for cloud IPs. Treat it as
        best-effort — it can rate-limit, return 5xx, or vanish. Configure
        TRANSCRIPT_PROVIDER_URL in your environment to swap providers.

        Returns (transcript_text, error_str).
        """
        try:
            base = DEFAULT_THIRDPARTY_URL.rstrip('/')
            url = f"{base}/videos"
            resp = requests.get(
                url,
                params={"part": "transcript", "id": video_id},
                timeout=10,
                headers={"User-Agent": "LearnTube-AI/1.0"},
            )
            if resp.status_code != 200:
                return "", f"thirdparty_status_{resp.status_code}"
            data = resp.json()
            items = data.get("items") or []
            if not items:
                return "", "thirdparty_no_items"
            transcript = items[0].get("transcript") or {}
            segments = transcript.get("items") or []
            parts = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                text = seg.get("text") or seg.get("transcriptText") or ""
                if text:
                    parts.append(text)
            if not parts:
                return "", "thirdparty_no_segments"
            return " ".join(parts), None
        except requests.RequestException as e:
            return "", f"thirdparty_request({e})"
        except Exception as e:
            return "", f"thirdparty({e})"

    @classmethod
    def extract_transcript(cls, youtube_url: str, manual_transcript: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract video transcript using, in order:
          1. user-pasted manual transcript (if provided)
          2. LangChain YoutubeLoader  (skipped when a proxy is configured,
             because the loader does not honor proxy env vars)
          3. youtube-transcript-api directly, optionally through YOUTUBE_PROXY
          4. third-party transcript provider (TRANSCRIPT_PROVIDER_URL)

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

        # 1. Manual transcript short-circuit.
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

        proxy_url = os.getenv('YOUTUBE_PROXY') or os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        use_proxy = bool(proxy_url and GenericProxyConfig)

        transcript_text = ""
        raw_docs: List[Any] = []
        last_error: Optional[str] = None

        # 2. LangChain loader (skipped when a proxy is configured, since the
        # loader ignores HTTP_PROXY env vars and would just hit the cloud IP).
        if not use_proxy:
            transcript_text, raw_docs, last_error = cls._fetch_via_langchain(youtube_url)
            if transcript_text and raw_docs and raw_docs[0].metadata and 'title' in raw_docs[0].metadata:
                video_title = raw_docs[0].metadata['title'] or video_title

        # 3. youtube-transcript-api direct (with proxy if configured).
        if not transcript_text.strip():
            try:
                transcript_text, err = cls._fetch_via_youtube_transcript_api(
                    video_id, proxy_url if use_proxy else None
                )
                if err:
                    last_error = err
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

        # 4. Third-party fallback for cloud deployments where YouTube directly blocks.
        if not transcript_text.strip():
            text, err = cls._fetch_via_third_party(video_id)
            if text:
                transcript_text = text
                # If we got the transcript from the third party, the YouTube IP
                # block is the most likely reason; annotate the error for logging.
                last_error = "thirdparty_recovered_after_ip_block"
            elif err:
                last_error = err

        cleaned_transcript = cls.clean_text(transcript_text)
        if not cleaned_transcript:
            if last_error == "ip_blocked" or (last_error and any(
                tok in last_error.lower() for tok in ("ipblocked", "requestblocked", "blocked", "bot")
            )):
                return {
                    'success': False,
                    'video_id': video_id,
                    'error': (
                        "YouTube blocked automatic transcript extraction from this server IP "
                        "and the third-party fallback also failed. Configure YOUTUBE_PROXY in "
                        "your environment, or paste the transcript text in the 'Manual "
                        "Transcript' box below to proceed."
                    )
                }
            if last_error:
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