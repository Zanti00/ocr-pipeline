from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ocr-pipeline"
    app_env: str = "development"
    app_port: int = 8010

    # API Keys for incoming requests
    serms_api_key: str
    prs_api_key: str
    
    # API Key for callbacks
    callback_api_key: str
    callback_max_retries: int = 3
    callback_backoff_seconds: str = "10,30,60"

    # Ollama config
    ollama_base_url: str
    ollama_model: str
    llm_provider: str = "ollama"

    # Databases
    redis_url: str
    mongodb_url: str
    mongodb_database: str
    postgres_url: str

    # Duplicate detection config
    embedding_model: str = "all-MiniLM-L6-v2"
    duplicate_similarity_threshold: float = 0.85
    duplicate_days_window: int = 90

    # OCR Config
    ocr_language: str = "eng"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
