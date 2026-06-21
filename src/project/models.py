"""Typed models for shared project infrastructure."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Project(BaseModel):
    """A validated reference to a behavior-suite project directory."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    name: str = Field(min_length=1)
