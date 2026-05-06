"""Солвер для matching с учётом предыдущей ошибки."""

import logging

from src.ai_client import AIClient
from src.exceptions import DOMElementNotFoundError
from src.stepik import (
    StepikAPIClient,
    StepData,
    AttemptData,
    try_solve_matching,
    build_ordering_reply,
    strip_html,
)

from .base import BaseSolver

logger = logging.getLogger(__name__)


class MatchingSolver(BaseSolver):

    async def solve(
        self,
        api: StepikAPIClient,
        ai: AIClient,
        step: StepData,
        attempt: AttemptData,
        previous_replies: list[dict] | None = None,
    ) -> dict:
        """Решает matching программно или через AI и возвращает ordering-reply."""
        logger.info(
            "Matching step_id=%d | dataset keys=%s",
            step.step_id, list(attempt.dataset.keys()),
        )
        logger.debug("dataset=%s", attempt.dataset)

        ordering = try_solve_matching(step, attempt)
        if ordering is not None:
            return build_ordering_reply(ordering)

        left, right = self._extract_lists(attempt)

        if not left or not right:
            raise DOMElementNotFoundError(
                f"Пустые данные matching. "
                f"keys={list(attempt.dataset.keys())}, "
                f"pairs={len(attempt.dataset.get('pairs', []))}"
            )

        question = step.question_text
        if previous_replies:
            prev_ordering = previous_replies[-1].get("ordering", [])
            question += (
                f"\n\nПредыдущий порядок {prev_ordering} был НЕВЕРНЫМ. "
                f"Попробуй другое сопоставление."
            )

        resp = await ai.solve_ordering_task(question, left, right)
        return build_ordering_reply(resp.ordered_indices)

    @staticmethod
    def _extract_lists(
        attempt: AttemptData,
    ) -> tuple[list[str], list[str]]:
        """Достает левый/правый списки из dataset с fallback по полям."""
        pairs = attempt.dataset.get("pairs", [])

        left = [strip_html(str(p.get("first", ""))) for p in pairs]

        # Правая часть: сначала пробуем options, потом pairs.second
        options = attempt.dataset.get("options", [])
        if options:
            right = [strip_html(str(o)) for o in options]
        else:
            right = [strip_html(str(p.get("second", ""))) for p in pairs]

        return left, right
