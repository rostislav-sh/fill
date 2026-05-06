"""Обработка одного шага — класс StepProcessor."""

import logging
import asyncio

from pydantic import BaseModel, ConfigDict

from src.ai_client import AIClient
from src.db.knowledge_cache import KnowledgeCache
from src.config import settings
from src.exceptions import (
    AIClientResponseError,
    InvalidAnswerIndicesError,
)
from src.stepik import (
    StepikAPIClient,
    StepikAPIError,
    StepikAuthError,
)

from .solver_registry import SolverRegistry
from .step_checker import StepChecker

logger = logging.getLogger(__name__)

_EXPECTED_ERRORS = (
    AIClientResponseError,
    InvalidAnswerIndicesError,
)


class ProcessStepResult(BaseModel):
    """Результат обработки одного шага."""
    model_config = ConfigDict(frozen=True)

    success: bool


class StepProcessor:
    """Решает шаг: проверка статуса, решение, отправка и проверка результата."""

    def __init__(
        self,
        ai: AIClient,
        api: StepikAPIClient,
        registry: SolverRegistry,
        max_attempts: int | None = None,
        delay: float | None = None,
        cache: KnowledgeCache | None = None,
    ) -> None:
        """Инициализирует зависимости и параметры retry на уровне шага."""
        self.ai = ai
        self.api = api
        self.registry = registry
        self.max_attempts = max_attempts or max(1, settings.step_solve_attempts)
        self.delay = delay or settings.api_delay_between_steps_sec
        self.cache = cache

        self.checker = StepChecker(api)

    async def process(self, step_id: int) -> ProcessStepResult:
        """Запускает полный pipeline обработки для одного `step_id`."""

        if await self.checker.is_already_passed(step_id):
            logger.info("Шаг %d уже решён ✔", step_id)
            return ProcessStepResult(success=True)

        try:
            step = await self.checker.get_step(step_id)
        except StepikAuthError:
            logger.error("Сессия истекла.")
            return ProcessStepResult(success=False)
        except StepikAPIError as exc:
            logger.error("Ошибка шага: %s", exc)
            return ProcessStepResult(success=False)

        logger.info(
            "═══ Шаг %d │ %s │ урок %d pos %d ═══",
            step_id, step.block_type,
            step.lesson_id, step.position,
        )

        if self.checker.should_skip(step):
            logger.info("Тип «%s» — пропуск.", step.block_type)
            return ProcessStepResult(success=True)

        solver = self.registry.get(step.block_type)
        if solver is None:
            logger.warning("Нет солвера для «%s».", step.block_type)
            return ProcessStepResult(success=True)

        return await self._attempt_loop(step_id, step, solver)

    async def _attempt_loop(self, step_id, step, solver):
        """Повторяет попытки решения шага до лимита или успеха."""
        cached_reply: dict | None = None
        if self.cache:
            cached_reply = await self.cache.get_reply(step_id, step.question_text)

        previous_replies: list[dict] = []
        for num in range(1, self.max_attempts + 1):
            try:
                attempt = await self.api.create_attempt(step_id)

                if cached_reply is not None:
                    reply = cached_reply
                    logger.info("[CACHE] Шаг %d попытка %d/%d (id=%d)", step_id, num, self.max_attempts, attempt.attempt_id)
                else:
                    reply = await solver.solve(
                        self.api, self.ai, step, attempt, previous_replies or None,
                    )
                    logger.info("[AI] Шаг %d попытка %d/%d (id=%d)", step_id, num, self.max_attempts, attempt.attempt_id)

                logger.debug("Reply: %s", reply)

                sub_id = await self.api.submit_answer(
                    attempt.attempt_id, reply,
                )
                status = await self.api.poll_status(sub_id)

                if status == "correct":
                    logger.info("✅ Шаг %d решён!", step_id)
                    if self.cache and cached_reply is None:
                        await self.cache.save_reply(
                            step_id, step.block_type, step.question_text, reply
                        )
                    return ProcessStepResult(success=True)

                if status == "wrong":
                    logger.warning("❌ Шаг %d неверно (%d/%d).", step_id, num, self.max_attempts)
                    previous_replies.append(reply)
                    if cached_reply is not None:
                        await self.cache.delete_reply(step_id)
                        cached_reply = None
                    if num < self.max_attempts:
                        await asyncio.sleep(self.delay)
                        continue
                    return ProcessStepResult(success=False)

                logger.warning("Статус «%s» — считаем успешным.", status)
                return ProcessStepResult(success=True)

            except _EXPECTED_ERRORS as exc:
                cached_reply = None
                logger.error("Попытка %d: %s", num, exc)
                if num < self.max_attempts:
                    continue
                return ProcessStepResult(success=False)

            except StepikAuthError:
                logger.error("Сессия API истекла.")
                return ProcessStepResult(success=False)

            except StepikAPIError as exc:
                logger.error("API: %s", exc)
                return ProcessStepResult(success=False)

        return ProcessStepResult(success=False)
