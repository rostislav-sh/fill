"""Абстрактный солвер."""

from abc import ABC, abstractmethod

from src.ai_client import AIClient
from src.stepik import StepikAPIClient, StepData, AttemptData


class BaseSolver(ABC):
    """Один солвер — один тип задания."""

    @abstractmethod
    async def solve(
        self,
        api: StepikAPIClient,
        ai: AIClient,
        step: StepData,
        attempt: AttemptData,
        previous_replies: list[dict] | None = None,
    ) -> dict:
        """Возвращает `reply`-словарь для отправки в Stepik API."""
        ...
