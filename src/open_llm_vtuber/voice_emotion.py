"""Voice emotion tags — deterministic per-sentence mapping (no LLM, no I/O).

The sentence pipeline already extracts semantic emotion keys per sentence
(``actions.emotions``, e.g. ``joy``/``sadness``/``anger``). The avatar uses
them for expressions; the voice path ignored them. This module maps those
keys onto ElevenLabs v3 conversational audio tags (``[happy]`` et al.),
prepended to the SYNTHESIS text only — display text, persisted history and
translation input are never touched.

Provider honesty: tag interpretation happens inside ElevenLabs' model and is
NOT contractually guaranteed (an unknown tag may be ignored or read aloud,
which is why the allowlist below is deliberately tiny and every unknown key
maps to None = plain synthesis). This layer only guarantees determinism:
same keys → same tag, stateless per sentence (no cross-turn leakage by
construction), fail-soft on any corrupt input.

Kill switch: ``ElevenLabsTTSConfig.emotion_tags_enabled`` (default True);
when False the engine reports plain synthesis for every sentence.
"""

from __future__ import annotations

from typing import Any, Optional

# Semantic emotion key (lowercased, as produced by extract_emotion_keys)
# -> ElevenLabs v3 audio tag. Fixed priority order = dict insertion order:
# the first matching key wins. Keys the extractor cannot produce
# (annoyed/assertive/caring/...) intentionally have NO entry: they synthesize
# plain rather than risk an unproven tag being read aloud.
EMOTION_AUDIO_TAGS = {
    "anger_strong": "angry",
    "anger": "angry",
    "fear": "fearful",
    "sadness": "sad",
    "disgust": "disgusted",
    "surprise": "surprised",
    "joy": "happy",
    "smirk": "playful",
    "embarrassed": "shy",
}


def emotion_tag_for(emotions: Any) -> Optional[str]:
    """Audio tag for one sentence's emotion keys, or None for plain voice.

    Pure, stateless, fail-soft: any non-list/corrupt input yields None.
    Matching is exact on lowercased keys — no substring guessing, so a key
    like ``enjoy`` can never accidentally match ``joy``.
    """
    try:
        if not isinstance(emotions, (list, tuple)):
            return None
        keys = set()
        for item in emotions:
            try:
                text = str(item or "").strip().lower()
            except Exception:
                continue
            if text:
                keys.add(text)
        if not keys:
            return None
        for key, tag in EMOTION_AUDIO_TAGS.items():
            if key in keys:
                return tag
        return None
    except Exception:
        return None


def tag_tts_text(text: str, tag: Optional[str]) -> str:
    """Prepend ``[tag]`` for synthesis, or return text unchanged.

    Never raises; empty/whitespace text is returned as-is (the silent-payload
    path in the TTS manager owns that case, not this helper).
    """
    try:
        if not tag or not str(text or "").strip():
            return text
        clean = str(tag).strip().strip("[]")
        if not clean:
            return text
        return f"[{clean}] {text}"
    except Exception:
        try:
            return text
        except Exception:
            return ""


__all__ = ["EMOTION_AUDIO_TAGS", "emotion_tag_for", "tag_tts_text"]
