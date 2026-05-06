from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

_env_path = Path(__file__).resolve().parents[1]/ ".env"


class Settings(BaseSettings):
    stepik_course_url: str = Field(default="https://stepik.org/catalog", validation_alias="STEPIK_COURSE_URL")
    stepik_email: str = Field(validation_alias="STEPIK_EMAIL")
    stepik_password: str = Field(validation_alias="STEPIK_PASSWORD")
    stepik_client_id: str = Field(validation_alias="STEPIK_CLIENT_ID")
    stepik_client_secret: str = Field(validation_alias="STEPIK_CLIENT_SECRET")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    fast_mode: bool = Field(default=True, validation_alias="FAST_MODE")
    step_solve_attempts: int = Field(default=2, validation_alias="STEP_SOLVE_ATTEMPTS")
    api_retry_attempts: int = Field(default=2, validation_alias="API_RETRY_ATTEMPTS")
    api_delay_between_steps_sec: float = Field(default=0.1, validation_alias="API_DELAY_BETWEEN_STEPS_SEC")

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", validation_alias="ANTHROPIC_MODEL")

    database_url: str = Field(default="", validation_alias="DATABASE_URL")

    ai_provider: str = Field(default="gemini", validation_alias="AI_PROVIDER")

    next_step_text: str = Field(default="Следующий шаг,Далее,Next step,Continue", validation_alias="NEXT_STEP_TEXTS")
    cookie_accept_texts: str = Field(default="Принять,Accept,Allow,Хорошо", validation_alias="ACCEPT_TEXTS")

    @model_validator(mode="after")
    def validate_ai_provider_settings(self) -> "Settings":
        """Нормализует `AI_PROVIDER` и проверяет обязательные ключи для выбранного режима."""
        provider = self.ai_provider.strip().lower()
        self.ai_provider = provider

        if provider not in {"gemini", "groq", "anthropic", "auto"}:
            raise ValueError("AI_PROVIDER должен быть одним из: gemini, groq, anthropic, auto.")

        if provider == "gemini" and not self.gemini_api_key.strip():
            raise ValueError("Для AI_PROVIDER=gemini требуется GEMINI_API_KEY.")

        if provider == "groq" and not self.groq_api_key.strip():
            raise ValueError("Для AI_PROVIDER=groq требуется GROQ_API_KEY.")

        if provider == "anthropic" and not self.anthropic_api_key.strip():
            raise ValueError("Для AI_PROVIDER=anthropic требуется ANTHROPIC_API_KEY.")

        if provider == "auto" and not (
            self.gemini_api_key.strip()
            or self.groq_api_key.strip()
            or self.anthropic_api_key.strip()
        ):
            raise ValueError(
                "Для AI_PROVIDER=auto требуется хотя бы один ключ: "
                "GEMINI_API_KEY, GROQ_API_KEY или ANTHROPIC_API_KEY."
            )

        return self

    model_config = SettingsConfigDict(
        env_file=str(_env_path) if _env_path.exists() else None,
        extra="ignore",
    )


settings = Settings()
