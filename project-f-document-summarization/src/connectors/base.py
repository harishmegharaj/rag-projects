"""Abstract integration boundary: fetch bytes + filename from any upstream system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPayload:
    """Normalized document for summarization."""

    filename: str
    data: bytes
    external_id: str | None = None


class DocumentSource(ABC):
    """Implement for S3, SharePoint, GDrive, CMS exports, etc."""

    @abstractmethod
    def fetch(self, ref: str) -> DocumentPayload:
        """Load document bytes by application-specific reference (URI, object key, drive id)."""
