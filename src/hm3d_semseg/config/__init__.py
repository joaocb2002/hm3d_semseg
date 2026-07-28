"""Strict project configuration loading."""

from hm3d_semseg.config.loader import load_config, save_resolved_config
from hm3d_semseg.config.schema import ProjectConfig

__all__ = ["ProjectConfig", "load_config", "save_resolved_config"]
