"""Core AI and Core ML deployment support for FlashVSR+."""

from .model import (
    CoreAIFirstChunkAttention,
    CoreAINextChunkAttention,
    build_rope_cos_sin,
    load_flashvsr_attention_weights,
)

__all__ = [
    "CoreAIFirstChunkAttention",
    "CoreAINextChunkAttention",
    "build_rope_cos_sin",
    "load_flashvsr_attention_weights",
]
