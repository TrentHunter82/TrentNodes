"""Lightweight types and constants for the H3 Hermes wire contract.

The HTTP boundary deliberately uses plain JSON-compatible containers.  These
small dataclasses type the parsed result without introducing pydantic (or any
other schema/runtime dependency) into ComfyUI startup.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

REQUEST_SCHEMA_VERSION = "h3_hermes_request/1.0"
RESPONSE_SCHEMA_VERSION = "h3_hermes_result/1.0"
TARGET_MODEL = "MiniMax H3"

H3_MODES: Tuple[str, ...] = (
    "ref",
    "base_T2VA",
    "base_I2VA",
    "base_FL2VA",
    "base_L2VA",
)
QUALITY_MODES: Tuple[str, ...] = ("fast", "balanced", "hero")
RESEARCH_POLICIES: Tuple[str, ...] = ("never", "when_uncertain", "always")

# These caps bound text crossing the agent boundary, not staged media.  Callers
# may lower them per deployment/test but cannot accidentally bypass checking by
# passing a byte string: the public contract is UTF-8 JSON text.
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = MAX_REQUEST_BYTES
DEFAULT_MAX_RESPONSE_BYTES = MAX_RESPONSE_BYTES

# Soft agent budgets.  The node's wall-clock timeout remains authoritative.
# Hero's two reviewers and two repairs are deliberately bounded; candidate
# count never exceeds the three named hero policies.
QUALITY_BUDGETS: Dict[str, Dict[str, int]] = {
    "fast": {
        "candidate_count": 1,
        "max_repairs": 0,
        "tool_call_target": 8,
        "subagent_target": 0,
        "max_subagents": 0,
    },
    "balanced": {
        "candidate_count": 2,
        "max_repairs": 1,
        "tool_call_target": 18,
        "subagent_target": 1,
        "max_subagents": 1,
    },
    "hero": {
        "candidate_count": 3,
        "max_repairs": 2,
        "tool_call_target": 32,
        "subagent_target": 2,
        "max_subagents": 2,
    },
}


@dataclass(frozen=True)
class HermesCandidate:
    """One model-proposed H3 prompt and its explicitly untrusted critique."""

    candidate_id: str
    policy: str
    prompt: str
    score_vector: Dict[str, Any]
    critic_findings: List[Any]

    @property
    def id(self) -> str:
        """Short alias useful in diagnostics and selection UIs."""

        return self.candidate_id


@dataclass(frozen=True)
class ParsedHermesResult:
    """Validated `h3_hermes_result/1.0` returned by :func:`parse_result`."""

    schema_version: str
    request_id: str
    status: str
    evidence: Dict[str, Any]
    intent_ir: Dict[str, Any]
    candidates: List[HermesCandidate]
    selected_candidate_id: str
    h3_prompt: str
    repairs: List[Any]
    quality_report: Dict[str, Any]
    reported_tools: List[Any]
    reported_sources: List[Any]

    @property
    def selected_id(self) -> str:
        return self.selected_candidate_id

    @property
    def prompt(self) -> str:
        return self.h3_prompt

    @property
    def selected_candidate(self) -> HermesCandidate:
        # Construction is only possible after the parser proved exactly one
        # matching candidate, so this cannot fail for a parsed result.
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )


# Compatibility aliases are intentionally cheap: callers can use the concise
# names while the wire format keeps its explicit Hermes wording.
Candidate = HermesCandidate
HermesResult = ParsedHermesResult
RESULT_SCHEMA_VERSION = RESPONSE_SCHEMA_VERSION
