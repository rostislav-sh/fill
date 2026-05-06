"""Солвер для sorting с учётом предыдущей ошибки."""

import logging

from src.ai_client import AIClient
from src.exceptions import DOMElementNotFoundError
from src.stepik import (
    StepikAPIClient,
    StepData,
    AttemptData,
    try_solve_sorting,
    build_ordering_reply,
    strip_html,
)

from .base import BaseSolver

logger = logging.getLogger(__name__)


class SortingSolver(BaseSolver):

    async def solve(
        self,
        api: StepikAPIClient,
        ai: AIClient,
        step: StepData,
        attempt: AttemptData,
        previous_replies: list[dict] | None = None,
    ) -> dict:
        """Решает sorting программно или через AI и формирует ordering-reply."""
        ordering = try_solve_sorting(step, attempt)
        if ordering is not None:
            return build_ordering_reply(ordering)

        items = self._extract_items(attempt)
        if not items:
            raise DOMElementNotFoundError("Пустые данные sorting.")

        question = step.question_text
        if previous_replies:
            previous_reply = previous_replies[-1]
            question += (
                f"\n\nПредыдущий порядок {previous_reply.get('ordering', [])} "
                f"был НЕВЕРНЫМ. Попробуй другой."
            )

        resp = await ai.solve_ordering_task(question, items, items)
        return build_ordering_reply(resp.ordered_indices)

    @staticmethod
    def _extract_items(attempt: AttemptData) -> list[str]:
        """Извлекает и очищает элементы sorting из dataset options."""
        options = attempt.dataset.get("options", [])
        return [strip_html(str(o)) for o in options]
