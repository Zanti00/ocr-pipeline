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

    async def choose_category(self, details: str, options: tuple[str, str]) -> str | None:
        """Break a tie between exactly two candidate categories."""
        return None
