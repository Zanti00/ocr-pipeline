import pytest

from app.core.anchors import AnchorScore
from app.core import ocr_engine
from app.core.ocr_engine import OcrReading, select_primary


def reading(
    text: str,
    score: float,
    confidence: float,
    *,
    engine: str = "tesseract",
    variant: str = "raw",
    psm: int = 6,
) -> OcrReading:
    return OcrReading(
        text=text,
        words=[],
        confidence=confidence,
        variant=variant,
        psm=psm,
        engine=engine,
        anchors=AnchorScore(total=score, mean_confidence=confidence),
        lines=[],
    )


def test_select_primary_prefers_anchor_score_over_confidence():
    paddle = reading(
        "Paddle total 10.00", 4.0, 0.99,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="contrast")

    assert select_primary([paddle, tesseract]) is tesseract


def test_select_primary_uses_confidence_as_tie_breaker():
    paddle = reading(
        "Paddle total 10.00", 5.0, 0.99,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 5.0, 0.88)

    assert select_primary([paddle, tesseract]) is paddle


def test_select_primary_rejects_empty_candidates():
    with pytest.raises(RuntimeError, match="empty"):
        select_primary([reading("   ", 10.0, 1.0)])


def test_read_pooled_can_select_tesseract_and_preserve_all_candidates(monkeypatch):
    paddle = reading(
        "Paddle total 10.00", 4.0, 0.99,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="contrast")
    alternate = reading("Other total 10.00", 5.5, 0.90, variant="flat", psm=4)

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    monkeypatch.setattr(
        ocr_engine,
        "read_best",
        lambda image, lang, psms: (tesseract, [tesseract, alternate]),
    )

    bundle = ocr_engine.read_pooled(object(), pool_size=3)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.all_readings == [paddle, tesseract, alternate]
    assert bundle.primary not in bundle.supporting
    assert len(bundle.supporting) == 2


def test_read_pooled_falls_back_to_tesseract_when_paddle_fails(monkeypatch):
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)

    def fail_paddle(image):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ocr_engine, "read_paddle", fail_paddle)
    monkeypatch.setattr(
        ocr_engine,
        "read_best",
        lambda image, lang, psms: (tesseract, [tesseract]),
    )

    bundle = ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.supporting == []


def test_read_pooled_does_not_duplicate_identical_text(monkeypatch):
    paddle = reading(
        "Same total 10.00", 5.0, 0.95,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Same total 10.00", 5.0, 0.90)
    alternate = reading("Different total 10.00", 4.0, 0.80, variant="flat")

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    monkeypatch.setattr(
        ocr_engine,
        "read_best",
        lambda image, lang, psms: (tesseract, [tesseract, alternate]),
    )

    bundle = ocr_engine.read_pooled(object(), pool_size=3)

    assert bundle.combined_text.count("Same total 10.00") == 1
    assert alternate in bundle.supporting
