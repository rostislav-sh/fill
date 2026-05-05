"""Точка входа."""

import asyncio
import logging

from src.db.knowledge_cache import KnowledgeCache
from src.config import settings
from src.logging_config import setup_logging
from src.ai_client import AIClient
from src.stepik.auth import StepikAuth
from src.stepik import StepikHTTPClient, StepikAPIClient
from src.stepik.utils import parse_course_id
from src.orchestration import (
    create_default_registry,
    StepProcessor,
    CourseProcessor,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Запускает полный pipeline: авторизация, обход курса и решение шагов."""
    setup_logging()

    course_url = settings.stepik_course_url
    if not course_url:
        logger.error("Укажите STEPIK_COURSE_URL в .env")
        return

    course_id = parse_course_id(course_url)
    if not course_id:
        logger.error("Не удалось извлечь ID курса: %s", course_url)
        return

    logger.info("Курс: %s (id=%d)", course_url, course_id)

    # Авторизация с автообновлением
    auth = StepikAuth(
        client_id=settings.stepik_client_id,
        client_secret=settings.stepik_client_secret,
        email=settings.stepik_email,
        password=settings.stepik_password,
    )

    cache: KnowledgeCache | None = None
    if settings.database_url:
        cache = KnowledgeCache(settings.database_url)
        await cache.init()
    else:
        logger.info("DATABASE_URL не задан. KnowledgeCache [off]")

    ai = AIClient()
    registry = create_default_registry()

    try:
        async with StepikHTTPClient(auth) as http:
            api = StepikAPIClient(http)
            step_proc = StepProcessor(ai, api, registry, cache=cache)
            course_proc = CourseProcessor(step_proc)

            solved = await course_proc.process_course(course_id)
            logger.info("🎓 Итого: %d шагов.", solved)
    finally:
        if cache:
            await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
