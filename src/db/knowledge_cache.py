import json
import logging
from datetime import datetime, timezone
import asyncpg

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS questions (
    step_id    BIGINT PRIMARY KEY,
    step_type  TEXT NOT NULL,
    question   TEXT NOT NULL,
    reply      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
)
"""


class KnowledgeCache:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
        logger.info("KnowledgeCache initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def get_reply(self, step_id: int, question: str) -> dict | None:
        row = await self._pool.fetchrow(
            "SELECT question, reply FROM questions WHERE step_id = $1", step_id
        )
        if row is None:
            return None

        if row["question"] != question:
            await self.delete_reply(step_id)
            return None

        try:
            return json.loads(row["reply"])
        except json.JSONDecodeError:
            logger.warning("KnowledgeCache: invalid JSON для step_id=%d", step_id)
            return None

    async def save_reply(
        self,
        step_id: int,
        step_type: str,
        question: str,
        reply: dict,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO questions (step_id, step_type, question, reply, created_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (step_id) DO UPDATE SET
                step_type  = EXCLUDED.step_type,
                question   = EXCLUDED.question,
                reply      = EXCLUDED.reply,
                created_at = EXCLUDED.created_at
            """,
            step_id,
            step_type,
            question,
            json.dumps(reply, ensure_ascii=False),
            datetime.now(timezone.utc),
        )
        logger.info("KnowledgeCache saved reply step_id=%d (%s)", step_id, step_type)

    async def delete_reply(self, step_id: int) -> None:
        await self._pool.execute(
            "DELETE FROM questions WHERE step_id = $1", step_id
        )
        logger.warning("KnowledgeCache: deleted reply step_id=%d", step_id)
