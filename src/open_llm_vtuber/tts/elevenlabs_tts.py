# src/open_llm_vtuber/tts/elevenlabs_tts.py
import os
from pathlib import Path

from loguru import logger
from elevenlabs.client import ElevenLabs

from .tts_interface import TTSInterface, TTSError, TTSQuotaExceededError


class TTSEngine(TTSInterface):
    """
    Uses ElevenLabs TTS API to generate speech.
    API Reference: https://elevenlabs.io/docs/api-reference/text-to-speech
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_v3_conversational",
        output_format: str = "mp3_44100_128",
        stability: float = 0.5,
        similarity_boost: float = 0.5,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        speed: float = 0.80,
        emotion_tags_enabled: bool = True,
    ):
        """
        Initializes the ElevenLabs TTS engine.

        Args:
            api_key (str): API key for ElevenLabs service.
            voice_id (str): Voice ID from ElevenLabs (e.g., JBFqnCBsd6RMkjVDRZzb).
            model_id (str): Model ID for ElevenLabs (e.g., eleven_multilingual_v2).
            output_format (str): Output audio format (e.g., mp3_44100_128).
            stability (float): Voice stability (0.0 to 1.0).
            similarity_boost (float): Voice similarity boost (0.0 to 1.0).
            style (float): Voice style exaggeration (0.0 to 1.0).
            use_speaker_boost (bool): Enable speaker boost for better quality.
            speed (float): Speech speed (0.7 to 1.2; 1.0 is normal speed).
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.speed = speed
        # Voice Emotion kill switch: when False, the TTS manager synthesizes
        # every sentence plain (no [tag] prefix), regardless of detected mood.
        self.emotion_tags_enabled = bool(emotion_tags_enabled)

        # Determine file extension from output format
        if "mp3" in output_format:
            self.file_extension = "mp3"
        elif "pcm" in output_format:
            self.file_extension = "wav"
        else:
            logger.warning(
                f"Unknown output format '{output_format}', defaulting to mp3 extension."
            )
            self.file_extension = "mp3"  # Default to mp3

        try:
            # Initialize ElevenLabs client
            self.client = ElevenLabs(api_key=api_key)
            logger.info("ElevenLabs TTS Engine initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize ElevenLabs client: {e}")
            self.client = None
            raise e

    def generate_audio(
        self, text: str, file_name_no_ext: str | None = None
    ) -> str | None:
        """
        Generate speech audio file using ElevenLabs TTS.

        Args:
            text (str): The text to synthesize.
            file_name_no_ext (str, optional): Name of the file without extension. Defaults to a generated name.

        Returns:
            str: The path to the generated audio file, or None if generation failed.
        """
        if not self.client:
            logger.error("ElevenLabs client not initialized. Cannot generate audio.")
            return None

        # Use the configured file extension
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)
        speech_file_path = Path(file_name)

        try:
            logger.debug(
                f"Generating audio via ElevenLabs for text: '{text[:50]}...' with voice '{self.voice_id}' model '{self.model_id}'"
            )

            # Eleven v4 models (e.g. eleven_v4_turbo) only support stability and similarity_boost.
            # Passing style, speed, or use_speaker_boost causes HTTP 400 Bad Request.
            is_v4 = bool(self.model_id and self.model_id.lower().startswith("eleven_v4"))
            voice_settings = {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            }
            if not is_v4:
                voice_settings["style"] = self.style
                voice_settings["use_speaker_boost"] = self.use_speaker_boost
                voice_settings["speed"] = self.speed

            # Generate audio using ElevenLabs API
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format,
                voice_settings=voice_settings,
            )

            # Write the audio data to file
            with open(speech_file_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            logger.info(
                f"Successfully generated audio file via ElevenLabs: {speech_file_path}"
            )

        except Exception as e:
            logger.critical(f"Error: ElevenLabs TTS unable to generate audio: {e}")
            # Clean up potentially incomplete file
            if speech_file_path.exists():
                try:
                    os.remove(speech_file_path)
                except OSError as rm_err:
                    logger.error(
                        f"Could not remove incomplete file {speech_file_path}: {rm_err}"
                    )

            if isinstance(e, TTSError):
                raise e

            status_code = getattr(e, "status_code", None)
            body = getattr(e, "body", None)
            detail = {}
            if isinstance(body, dict):
                detail = (
                    body.get("detail", {})
                    if isinstance(body.get("detail"), dict)
                    else body
                )

            code = detail.get("code") or detail.get("status") if isinstance(detail, dict) else None
            msg = detail.get("message") if isinstance(detail, dict) else (str(body) if body else str(e))
            msg_str = str(msg or e)
            body_str = str(body or "").lower()

            # 1. Quota Exceeded (specifically checking code/status/message, even with HTTP 401)
            if (
                code == "quota_exceeded"
                or "quota_exceeded" in body_str
                or "exceeds your quota" in body_str
                or "exceeds your quota" in msg_str.lower()
                or (status_code == 402)
            ):
                raise TTSQuotaExceededError(
                    msg_str,
                    http_status=status_code or 401,
                    original_error=e,
                ) from e
            elif status_code == 401:
                raise TTSError(
                    msg_str,
                    error_code=code or "authentication_failed",
                    error_category="authentication",
                    http_status=401,
                    retryable=False,
                    original_error=e,
                ) from e
            elif status_code == 403:
                raise TTSError(
                    msg_str,
                    error_code=code or "permission_denied",
                    error_category="permission",
                    http_status=403,
                    retryable=False,
                    original_error=e,
                ) from e
            elif status_code == 404:
                raise TTSError(
                    msg_str,
                    error_code=code or "not_found",
                    error_category="not_found",
                    http_status=404,
                    retryable=False,
                    original_error=e,
                ) from e
            elif status_code == 400:
                raise TTSError(
                    msg_str,
                    error_code=code or "invalid_request",
                    error_category="invalid_request",
                    http_status=400,
                    retryable=False,
                    original_error=e,
                ) from e
            elif status_code == 429:
                raise TTSError(
                    msg_str,
                    error_code=code or "rate_limited",
                    error_category="rate_limited",
                    http_status=429,
                    retryable=False,
                    original_error=e,
                ) from e
            else:
                raise TTSError(
                    str(e),
                    error_code="synthesis_failed",
                    error_category="unknown",
                    http_status=status_code,
                    retryable=False,
                    original_error=e,
                ) from e

        return str(speech_file_path)


# Example usage (optional, for testing)
# if __name__ == '__main__':
#     tts_engine = TTSEngine(
#         api_key="your_api_key",
#         voice_id="JBFqnCBsd6RMkjVDRZzb",
#         model_id="eleven_multilingual_v2"
#     )
#     test_text = "Hello world! This is a test using ElevenLabs."
#     audio_path = tts_engine.generate_audio(test_text, "elevenlabs_test")
#     if audio_path:
#         print(f"Generated audio saved to: {audio_path}")
#     else:
#         print("Failed to generate audio.")
