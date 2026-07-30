import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.prompts import (
    CATEGORY_TIEBREAK_PROMPT,
    FINANCIAL_SEMANTICS_PROMPT,
    ITEM_ANALYSIS_PROMPT,
    LOCATION_SELECTION_PROMPT,
    RECEIPT_EXTRACTION_PROMPT,
    VENDOR_SELECTION_PROMPT,
)


logger = logging.getLogger(__name__)

# Selection answers are a few tokens. Capping output keeps CPU inference short and
# stops a small model from rambling past valid JSON.
SELECTION_TOKEN_LIMIT = 64
EXTRACTION_TOKEN_LIMIT = 768


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    async def _generate(
        self, prompt: str, token_limit: int, timeout: float = 120.0
    ) -> dict[str, Any] | None:
        """Call Ollama and parse JSON, with one repair attempt.

        A single stray token used to fail the whole job: the old implementation let
        ``json.loads`` raise straight through into a failed callback, losing an
        otherwise good extraction.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": token_limit,
            },
        }

        async with httpx.AsyncClient() as client:
            for attempt in (1, 2):
                try:
                    response = await client.post(
                        f"{self.base_url}/api/generate", json=payload, timeout=timeout
                    )
                    response.raise_for_status()
                    raw = response.json().get("response", "")
                    return self._parse_json(raw)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Ollama returned unparseable JSON (attempt %s): %s",
                                   attempt, exc)
                    if attempt == 2:
                        return None
                except httpx.HTTPError as exc:
                    logger.warning("Ollama request failed (attempt %s): %s", attempt, exc)
                    if attempt == 2:
                        return None
        return None

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Salvage the first JSON object if the model wrapped it in prose.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise
            parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed

    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        result = await self._generate(
            RECEIPT_EXTRACTION_PROMPT.format(ocr_text=ocr_text),
            token_limit=EXTRACTION_TOKEN_LIMIT,
        )
        if result is None:
            raise RuntimeError("Ollama extraction failed after retry")
        return result

    async def select_vendor_name(
        self, candidates: list[str], excluded: list[str] | None = None
    ) -> str | None:
        if not candidates:
            return None
        prompt = VENDOR_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates),
            excluded="\n".join(f"- {line}" for line in (excluded or [])) or "- (none)",
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("vendor_name")
        return str(choice).strip() if choice else None

    async def select_location(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        prompt = LOCATION_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates)
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("location")
        return str(choice).strip() if choice else None

    async def choose_category(self, details: str, options: tuple[str, str]) -> str | None:
        prompt = CATEGORY_TIEBREAK_PROMPT.format(
            option_a=options[0], option_b=options[1], details=details[:1200]
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("category")
        if not choice:
            return None
        picked = str(choice).strip()
        # Reject anything outside the two options offered.
        return picked if picked in options else None

    async def analyze_line_items(self, lines: list[str]) -> list[dict] | None:
        if not lines:
            return None
        prompt = ITEM_ANALYSIS_PROMPT.format(
            lines="\n".join(f"- {line}" for line in lines[:50])
        )
        result = await self._generate(prompt, token_limit=EXTRACTION_TOKEN_LIMIT, timeout=60.0)
        if not result or "items" not in result:
            return None
        items = result.get("items")
        return items if isinstance(items, list) else None

    async def analyze_financial_semantics(self, ocr_text: str) -> dict | None:
        """Return a bounded semantics answer; amounts are prohibited by prompt."""
        if not ocr_text.strip():
            return None
        return await self._generate(
            FINANCIAL_SEMANTICS_PROMPT.format(ocr_text=ocr_text[:12000]),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=60.0,
        )

    async def verify_subtotal(self, ocr_text: str) -> float | None:
        if not ocr_text.strip():
            return None
        from app.llm.prompts import SUBTOTAL_VERIFICATION_PROMPT
        result = await self._generate(
            SUBTOTAL_VERIFICATION_PROMPT.format(ocr_text=ocr_text[:12000]),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=60.0,
        )
        if not result:
            return None
        choice = result.get("subtotal")
        try:
            return float(choice) if choice is not None else None
        except (ValueError, TypeError):
            return None
