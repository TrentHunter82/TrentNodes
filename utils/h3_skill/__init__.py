"""
Skill-driven MiniMax H3 prompt generation.

This package powers the H3 Skill Promptor node. Its system prompt is
the h3-prompting skill text itself (live file first, vendored snapshot
as fallback), its validation is a read-only checklist derived from the
skill's review list, and its transport is any OpenAI-compatible local
server (normally the managed llama-server from utils.llamacpp_server).

Deliberately independent of the h3_prompt/h3_cowboy generation
pipelines: only pure constants and the official MiniMax examples are
imported from them, never their assemblers or backends.
"""

from .skill_loader import (  # noqa: F401
    CHECKPOINT_FOR_MODE,
    MODES,
    build_system_prompt,
    build_user_context,
    load_skill_text,
)
from .checklist import validate, final_shot_index, assemble_final  # noqa: F401
from .client import build_user_message, chat  # noqa: F401
