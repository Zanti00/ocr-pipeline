from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider interface for the narrowed model role.

    Only ``extract_receipt_fields`` is abstract, so existing providers keep
    working. The selection methods have safe defaults that return ``None``,
    meaning "no opinion" - callers fall back to the deterministic result rather
    than failing, which keeps the model an optional upgrade instead of a hard
    dependency.
    """

    @abstractmethod
    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        """Legacy single-shot extraction from raw OCR text."""

    async def select_vendor_name(
        self, candidates: list[str], excluded: list[str] | None = None
    ) -> str | None:
        """Pick the issuing business from a closed candidate list.

        Returning ``None`` means no opinion; the caller keeps its own ranking.
        """
        return None

    async def select_location(self, candidates: list[str]) -> str | None:
        """Pick the business location/address from a closed candidate list.

        Returning ``None`` means no opinion; caller keeps deterministic result.
        """
        return None

    async def choose_category(self, details: str, options: tuple[str, str]) -> str | None:
        """Break a tie between exactly two candidate categories."""
        return None

    async def analyze_line_items(self, lines: list[str]) -> list[dict] | None:
        """Analyze line-item structures (quantity, name, price) from receipt lines."""
        return None

    async def analyze_financial_semantics(self, ocr_text: str) -> dict | None:
        """Classify tax basis/currency context without extracting any amounts."""
        return None

    async def verify_subtotal(self, ocr_text: str) -> float | None:
        """Extract the explicitly printed subtotal (net sales before tax) from the text."""
        return None

    async def normalize_text(self, text: str, context: str) -> str:
        """Correct spelling mistakes in a specific text snippet using full receipt context."""
        return text
