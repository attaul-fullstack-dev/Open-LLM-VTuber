import abc
import os
import asyncio
from typing import Optional

from loguru import logger


class TTSError(Exception):
    """Base structured exception for TTS synthesis failures."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "unknown",
        error_category: str = "unknown",
        http_status: Optional[int] = None,
        retryable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.error_category = error_category
        self.http_status = http_status
        self.retryable = retryable
        self.original_error = original_error

    def to_dict(self, provider: str = "") -> dict:
        return {
            "provider": provider,
            "code": self.error_code,
            "category": self.error_category,
            "message": str(self),
            "http_status": self.http_status,
            "retryable": self.retryable,
        }


class TTSQuotaExceededError(TTSError):
    """Raised when TTS provider quota or credits have been exceeded."""

    def __init__(
        self,
        message: str,
        *,
        http_status: Optional[int] = 401,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            error_code="quota_exceeded",
            error_category="quota_exceeded",
            http_status=http_status,
            retryable=False,
            original_error=original_error,
        )


class TTSInterface(metaclass=abc.ABCMeta):
    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Asynchronously generate speech audio file using TTS.

        By default, this runs the synchronous generate_audio in a coroutine.
        Subclasses can override this method to provide true async implementation.

        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext)

    @abc.abstractmethod
    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        raise NotImplementedError

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """
        Remove a file from the file system.

        This is a separate method instead of a part of the `play_audio_file_local()` because `play_audio_file_local()` is not the only way to play audio files. This method will be used to remove the audio file after it has been played.

        Parameters:
            filepath (str): The path to the file to remove.
            verbose (bool): If True, print messages to the console.
        """
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist")
            return
        try:
            logger.debug(f"Removing file {filepath}") if verbose else None
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to remove file {filepath}: {e}")

    def generate_cache_file_name(self, file_name_no_ext=None, file_extension="wav"):
        """
        Generate a cross-platform cache file name.

        file_name_no_ext: str
            name of the file without extension
        file_extension: str
            file extension

        Returns:
        str: the path to the generated cache file
        """
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if file_name_no_ext is None:
            file_name_no_ext = "temp"

        file_name = f"{file_name_no_ext}.{file_extension}"
        return os.path.join(cache_dir, file_name)
