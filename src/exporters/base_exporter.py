from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..models import CNCProject


OutputType = TypeVar("OutputType")


class BaseExporter(ABC, Generic[OutputType]):
    @abstractmethod
    def generate(self, project: CNCProject) -> OutputType:
        """Gera a saída em memória sem transmitir para equipamento algum."""

