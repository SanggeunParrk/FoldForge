"""Shared geometric guidance and constraint projection for diffusion."""

from .config import TFGConfig, build_terms, parse_tfg_config
from .engine import GuidanceEngine

__all__ = ["GuidanceEngine", "TFGConfig", "build_terms", "parse_tfg_config"]
