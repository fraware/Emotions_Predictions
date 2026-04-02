from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration; override with environment variables (prefix `EMOTIONS_`)."""

    model_config = SettingsConfigDict(
        env_prefix="EMOTIONS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: Path = Field(default=Path("EEG-emotions.csv"))
    output_dir: Path = Field(default=Path("outputs"))
    random_seed: int = 48

    test_fraction: float = Field(default=0.15, ge=0.05, le=0.4)
    val_fraction: float = Field(default=0.15, ge=0.05, le=0.4)

    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)

    gru_units: int = Field(default=256, ge=1)
    dropout_rate: float = Field(default=0.2, ge=0, lt=1)

    tensorboard_logdir: Path | None = None
