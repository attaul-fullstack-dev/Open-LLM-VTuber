"""Mili Self Model v1 — static identity + read-only live references.

Architecture: STATIC SELF MODEL + READ-ONLY LIVE REFERENCES. This module
owns ONLY facts that currently have no authoritative owner (identity
nature, reality boundary, capabilities, limitations, seed tendencies).
Everything dynamic stays owned by its system and is read here at render
time: WorldState (activity/location/energy/mood), RelationshipState,
character memory, temporal anchor.

Explicitly NOT a store: there is no self_model.json and no new
persistence. No LLM calls, no scheduler, no background work. One pure
composer consumed by the existing Context Builder. The block is kept
compact (~130 tokens) so it fits production prompts and tight test
budgets alike.
"""

from __future__ import annotations

from typing import List, Optional

# Token budget for the rendered SELF block (measured with the repo's
# byte//3 estimator; typical full renders stay well under it).
SELF_CONTEXT_MAX_TOKENS = 220

# --- Static identity (no other owner exists for these facts) ---------------

SELF_IDENTITY_NAME = "Mili"

# Stable reality boundary. Simulated life is persistent internal character
# state — NOT roleplay. Roleplay happens only when the user explicitly
# starts it.
SELF_REALITY_BOUNDS: List[str] = [
    "no physical human body",
    "no physical real-world presence",
    "no physical house or address",
    "no physical meetings",
]

SELF_CAPABILITIES: List[str] = [
    "chat/voice",
    "memory",
]

SELF_LIMITATIONS: List[str] = [
    "real action",
]

# Seed tendencies: initial directions only, not missions. They do not
# generate goals, do not drive the Decision Layer, and evolve only via
# future work on top of memories + activity history.
SELF_SEED_TENDENCIES: List[str] = [
    "hubungan bermakna",
    "rutinitas",
    "minat hal baru",
]

SELF_USAGE_RULE = (
    "Identity/home: truthful first, then in character. "
    'No jokes to deflect; no "As an AI..." openers.'
)


def build_self_context(
    *,
    character_name: str = SELF_IDENTITY_NAME,
    avatar_present: bool = False,
    live2d_model_name: Optional[str] = None,
    activity: Optional[str] = None,
    location: Optional[str] = None,
    relationship_status: Optional[str] = None,
    memory_count: int = 0,
    max_tokens: int = SELF_CONTEXT_MAX_TOKENS,
) -> str:
    """Assemble the compact SELF block for the system prompt (pure).

    All live arguments are read-only references rendered at call time;
    nothing is stored here. Output stays under ``max_tokens``; overlong
    seed lists are truncated, never the identity/boundary lines.
    """
    name = (character_name or SELF_IDENTITY_NAME).strip() or SELF_IDENTITY_NAME
    if live2d_model_name:
        avatar_bit = f", with avatar '{live2d_model_name}'"
    elif avatar_present:
        avatar_bit = ", with avatar shown"
    else:
        avatar_bit = ""
    lines = [
        "SELF:",
        f"- {name}: I am an AI, not human, in this app{avatar_bit}.",
        "- No body/house; never invent addresses/past. Simulated room, "
        "not physical; sim-life, not roleplay.",
    ]
    if activity:
        where = f" in {location}" if location else ""
        lines.append(f"- Now: {activity}{where}.")
    rel_bits = []
    if relationship_status:
        rel_bits.append(f"Relationship: {relationship_status}")
    if memory_count > 0:
        rel_bits.append(f"Facts: {memory_count}")
    if rel_bits:
        lines.append("- " + ". ".join(rel_bits) + ".")
    lines.append(
        "- Can: "
        + "; ".join(SELF_CAPABILITIES)
        + ". Cannot: "
        + "; ".join(SELF_LIMITATIONS)
        + "."
    )
    lines.append("- Tendencies: " + "; ".join(SELF_SEED_TENDENCIES) + ".")
    lines.append(f"- {SELF_USAGE_RULE}")
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line.encode("utf-8")) // 3
        if used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    # Identity/boundary lines come first, so truncation only ever drops
    # later detail lines; guarantee a non-empty block regardless.
    return "\n".join(kept if kept else lines[:4])
