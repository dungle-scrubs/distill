"""Saccade aliases for shared media-ingest warning and fatal error helpers."""

from __future__ import annotations

from .media.errors import MediaIngestError, WarningPayload, warning

SaccadeError = MediaIngestError

__all__ = ["SaccadeError", "WarningPayload", "warning"]
