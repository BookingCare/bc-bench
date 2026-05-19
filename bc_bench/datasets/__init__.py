"""Dataset registry."""

from __future__ import annotations

from ..types import PromptConfig

_PROMPT_CONFIGS: dict[str, PromptConfig] = {}


def register(name: str, prompt_config: PromptConfig) -> None:
    _PROMPT_CONFIGS[name] = prompt_config


def get_prompt_config(name: str) -> PromptConfig:
    return _PROMPT_CONFIGS[name]


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_PROMPT_CONFIGS))
