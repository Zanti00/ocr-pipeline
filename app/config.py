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
    # How long the model stays resident in memory after a call. -1 keeps the
    # model permanently resident in memory to eliminate cold reload latency.
    ollama_keep_alive: str = "-1"
    # Explicit context window (4096 gives >2x headroom over the 6000-char prompt cap
    # while halving KV cache allocation overhead).
    ollama_num_ctx: int = 4096
    # Timeout for optional LLM assists; fail soft to deterministic baseline
    ollama_assist_timeout: float = 20.0
    # Line items LLM normalization toggle (skips redundant 20-85s batch LLM call on CPU)
    enable_llm_item_normalization: bool = False

    # Cache TTLs (seconds)
    file_cache_byte_ttl: int = 600          # 10 minutes ephemeral byte handoff
    file_cache_result_ttl: int = 7776000     # 90 days file-hash result cache

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

    # OCR pool size tuning: which Tesseract variants and PSM modes the escalated
    # pool runs. Comma-separated lists; variants may be "all" for the full set.
    # Used by read_pooled/read_best - shrinking these speeds up hard receipts at
    # the cost of fewer candidate readings to choose from.
    ocr_pool_variants: str = "all"
    ocr_pool_psms: str = "6,4,11"

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
    image_quality_check_not_receipt: bool = True
    image_quality_max_face_area_ratio: float = 0.04
    image_quality_max_saturation_mean: float = 110.0
    image_quality_min_text_edge_density: float = 0.015

    # Rate limiting config
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_ocr_per_minute: int = 30
    rate_limit_duplicate_per_minute: int = 60
    rate_limit_read_per_minute: int = 120
    rate_limit_health_per_minute: int = 60
    rate_limit_auth_fail_per_minute: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
