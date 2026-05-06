"""Солвер для choice: сначала программно, потом AI."""

import logging

from src.ai_client import AIClient
from src.exceptions import DOMElementNotFoundError
from src.stepik import (
    StepikAPIClient,
    StepData,
    AttemptData,
    build_choice_reply,
    strip_html,
)
from src.stepik.solvers import try_solve_choice

from .base import BaseSolver

logger = logging.getLogger(__name__)


class ChoiceSolver(BaseSolver):

    async def solve(
        self,
        api: StepikAPIClient,
        ai: AIClient,
        step: StepData,
        attempt: AttemptData,
        previous_replies: list[dict] | None = None,
    ) -> dict:
        """Решает choice: программно → AI."""
        raw_opts = (
            step.source.options
            or attempt.dataset.get("options", [])
        )
        texts = self._extract_texts(raw_opts)

        if not texts:
            raise DOMElementNotFoundError(
                "Варианты ответов не найдены."
            )

        # 1. Попытка программного решения
        #    (только если это первая попытка)
        if not previous_replies:
            selected = try_solve_choice(step)
            if selected is not None:
                return build_choice_reply(selected, len(raw_opts))

        # 2. AI
        hint = (
            " (можно выбрать несколько)"
            if step.is_multiple_choice
            else " (выберите один)"
        )

        question = step.question_text + hint

        if previous_replies:
            if step.is_multiple_choice:
                tried_combinations = [
                    [texts[i] for i, v in enumerate(prev.get("choices", [])) if v and i < len(texts)]
                    for prev in previous_replies
                ]
                question += (
                    f"\n\nВНИМАНИЕ: следующие комбинации были НЕВЕРНЫМИ: {tried_combinations}. "
                    f"Попробуй другую комбинацию вариантов."
                )
                logger.info("Повторная попытка: исключаем комбинации %s", tried_combinations)
            else:
                wrong_texts: list[str] = []
                for prev in previous_replies:
                    for i, v in enumerate(prev.get("choices", [])):
                        if v and i < len(texts) and texts[i] not in wrong_texts:
                            wrong_texts.append(texts[i])
                question += (
                    f"\n\nВНИМАНИЕ: все предыдущие ответы были НЕВЕРНЫМИ. "
                    f"Неправильные варианты: {wrong_texts}. "
                    f"Выбери ДРУГОЙ ответ."
                )
                logger.info("Повторная попытка: исключаем %s", wrong_texts)

        resp = await ai.solve_choice_task(question, texts)
        return build_choice_reply(
            resp.selected_indices, len(raw_opts),
        )

    @staticmethod
    def _extract_texts(raw_opts: list) -> list[str]:
        texts: list[str] = []
        for o in raw_opts:
            if isinstance(o, dict):
                texts.append(strip_html(o.get("text", "")))
            else:
                texts.append(strip_html(str(o)))
        return [t for t in texts if t]
