"""Unit tests for the duplicate detection feature and SERMS callback payload integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.api.schemas.duplicate import DuplicateCheckRequest, DuplicateCheckResponse, DuplicateMatch
from app.api.schemas.ocr import build_callback_payload, OcrCallbackPayload


class TestDuplicateSchemas:
    def test_duplicate_callback_payload_defaults(self):
        payload = build_callback_payload(receipt_id=42, confidence=0.88)
        assert payload.is_duplicate is False
        assert payload.duplicate_similarity is None

    def test_duplicate_callback_payload_populated(self):
        payload = build_callback_payload(
            receipt_id=42,
            confidence=0.88,
            is_duplicate=True,
            duplicate_similarity=0.94238
        )
        assert payload.is_duplicate is True
        assert payload.duplicate_similarity == 0.9424

    def test_duplicate_check_request_response_models(self):
        req = DuplicateCheckRequest(
            receipt_text="Starbucks Coffee PHP 250.00",
            source_service="serms",
            threshold=0.85,
            days_window=90
        )
        assert req.threshold == 0.85

        match = DuplicateMatch(
            receipt_id=101,
            source_service="serms",
            similarity_score=0.92,
            processed_at="2026-07-29T12:00:00Z"
        )
        resp = DuplicateCheckResponse(is_duplicate=True, matches=[match])
        assert resp.is_duplicate is True
        assert len(resp.matches) == 1
        assert resp.matches[0].receipt_id == 101


@pytest.mark.asyncio
class TestDuplicateRouteHandler:
    @patch("app.api.routes.duplicate.EmbeddingGenerator.generate")
    @patch("app.api.routes.duplicate.find_similar_receipts", new_callable=AsyncMock)
    async def test_check_duplicate_route_success(self, mock_find, mock_generate):
        from app.api.routes.duplicate import check_duplicate

        mock_generate.return_value = [0.1] * 384
        mock_find.return_value = [
            DuplicateMatch(
                receipt_id=202,
                source_service="serms",
                similarity_score=0.89,
                processed_at="2026-07-29T10:00:00Z"
            )
        ]

        req = DuplicateCheckRequest(
            receipt_text="Shell Gas Station 1800 PHP",
            source_service="serms"
        )
        res = await check_duplicate(req, source_service="serms")

        assert res.is_duplicate is True
        assert len(res.matches) == 1
        assert res.matches[0].receipt_id == 202
        mock_generate.assert_called_once_with("Shell Gas Station 1800 PHP")

    @patch("app.api.routes.duplicate.EmbeddingGenerator.generate")
    async def test_check_duplicate_empty_text(self, mock_generate):
        from app.api.routes.duplicate import check_duplicate

        req = DuplicateCheckRequest(
            receipt_text="   ",
            source_service="serms"
        )
        res = await check_duplicate(req, source_service="serms")

        assert res.is_duplicate is False
        assert res.matches == []
        mock_generate.assert_not_called()
