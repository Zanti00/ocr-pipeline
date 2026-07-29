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

    # Image quality gate settings
    image_quality_blur_threshold: float = 80.0
    image_quality_brightness_floor: float = 40.0
    # Orientation-independent size floors. Thermal receipts are often narrow
    # (short side ~200-300px) but still OCR cleanly; require both sides.
    image_quality_min_short_side: int = 200
    image_quality_min_long_side: int = 400
    # Legacy aliases kept so existing IMAGE_QUALITY_MIN_WIDTH/HEIGHT env vars
    # still load without error (values are ignored; short/long sides are used).
    image_quality_min_width: int = 200
    image_quality_min_height: int = 400
    image_quality_max_segments: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
