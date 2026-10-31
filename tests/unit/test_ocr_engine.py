import pytest

from app.core import ocr_engine
from app.core.anchors import AnchorScore
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


@pytest.mark.asyncio
async def test_read_pooled_can_select_tesseract_and_preserve_all_candidates(monkeypatch):
    paddle = reading(
        "Paddle total 10.00", 4.0, 0.99,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="clean")

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": tesseract)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.all_readings == [paddle, tesseract]
    assert bundle.primary not in bundle.supporting
    assert bundle.supporting == [paddle]


@pytest.mark.asyncio
async def test_read_pooled_early_exits_when_paddle_scores_high(monkeypatch):
    paddle = reading(
        "Paddle total 10.00", 7.5, 0.90,
        engine="paddle", variant="source", psm=0,
    )
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)

    def mock_read_fast(*args, **kwargs):
        raise AssertionError("Tesseract must not run when Paddle fast path satisfies")

    monkeypatch.setattr(ocr_engine, "read_fast", mock_read_fast)

    bundle = await ocr_engine.read_pooled(object())

    assert bundle.early_exit is True
    assert bundle.primary is paddle
    assert bundle.all_readings == [paddle]


@pytest.mark.asyncio
async def test_read_pooled_escalates_when_paddle_scores_low(monkeypatch):
    weak_paddle = reading(
        "Weak noise", 2.0, 0.30,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="clean")
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: weak_paddle)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": tesseract)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.early_exit is False
    assert bundle.primary is tesseract
    assert len(bundle.all_readings) == 2


@pytest.mark.asyncio
async def test_read_pooled_falls_back_to_tesseract_when_paddle_fails(monkeypatch):
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="clean")

    def fail_paddle(image):
        raise RuntimeError("Paddle model unavailable")

    monkeypatch.setattr(ocr_engine, "read_paddle", fail_paddle)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": tesseract)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.all_readings == [tesseract]
    assert bundle.supporting == []


def test_fast_path_eligible_threshold():
    assert ocr_engine._fast_path_eligible(reading("x", ocr_engine.FAST_PATH_ANCHOR_SCORE, 0.9))
    assert not ocr_engine._fast_path_eligible(
        reading("x", ocr_engine.FAST_PATH_ANCHOR_SCORE - 0.01, 0.9)
    )


@pytest.mark.asyncio
async def test_read_pooled_does_not_duplicate_identical_text(monkeypatch):
    paddle = reading(
        "Same total 10.00", 5.0, 0.95,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Same total 10.00", 5.0, 0.90, variant="clean")

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": tesseract)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.combined_text.count("Same total 10.00") == 1
    assert bundle.supporting == []


def test_read_paddle_caches_reading_on_image():
    from PIL import Image

    call_count = 0

    class FakePredictEngine:
        def predict(self, img_array, **kwargs):
            nonlocal call_count
            call_count += 1
            return [{
                "rec_texts": ["Total 10.00"],
                "rec_scores": [0.99],
                "rec_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }]

    img = Image.new("RGB", (200, 200), color="white")
    orig_engine = ocr_engine._get_paddle_ocr
    try:
        ocr_engine._get_paddle_ocr = lambda: FakePredictEngine()
        res1 = ocr_engine.read_paddle(img)
        res2 = ocr_engine.read_paddle(img)
        assert res1 is res2
        assert call_count == 1
    finally:
        ocr_engine._get_paddle_ocr = orig_engine


def test_supporting_readings_drop_far_weaker_candidates():
    # Tall/narrow receipts: Paddle reads cleanly while Tesseract degrades to
    # near-zero scores. That garbage must not reach combined_text, which feeds
    # the LLM prompt and the item scanners.
    from app.core.ocr_engine import _supporting_readings

    primary = reading(
        "Paddle total 10.00", 4.46, 0.95,
        engine="paddle", variant="source", psm=0,
    )
    close = reading("Close runner total 10.00", 4.20, 0.90, variant="flat")
    garbage = reading("garbage !!! x", 0.36, 0.30, variant="raw")

    supporting = _supporting_readings([primary, close, garbage], primary, pool_size=3)

    assert supporting == [close]
    assert all(r.score >= primary.score / 2.0 for r in supporting)


def test_supporting_readings_keep_strong_runner_ups():
    from app.core.ocr_engine import _supporting_readings

    primary = reading("Paddle total 10.00", 4.0, 0.95, engine="paddle", variant="source", psm=0)
    runner = reading("Runner total 10.00", 3.6, 0.90, variant="flat")

    supporting = _supporting_readings([primary, runner], primary, pool_size=2)

    assert supporting == [runner]


def test_read_paddle_sets_det_limit_on_tall_receipts(monkeypatch):
    from PIL import Image
    captured = {}

    class FakePredictEngine:
        def predict(self, img_array, **kwargs):
            captured["kwargs"] = kwargs
            return [{
                "rec_texts": ["Total 10.00"],
                "rec_scores": [0.99],
                "rec_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }]

    monkeypatch.setattr(ocr_engine, "_get_paddle_ocr", lambda: FakePredictEngine())

    # Tall portrait receipt: width=200, height=500 -> aspect height >= 2 * width
    tall_img = Image.new("RGB", (200, 500), color="white")
    ocr_engine.read_paddle(tall_img)
    assert captured["kwargs"].get("text_det_limit_side_len") == 512

    # Square receipt: width=400, height=400 -> det_limit not set
    square_img = Image.new("RGB", (400, 400), color="white")
    ocr_engine.read_paddle(square_img)
    assert "text_det_limit_side_len" not in captured["kwargs"]


def test_read_paddle_uses_correct_orientation(monkeypatch):
    from PIL import Image
    captured = {}

    class FakePredictEngine:
        def predict(self, img_array, **kwargs):
            captured["shape"] = img_array.shape
            return [{
                "rec_texts": ["Total 10.00"],
                "rec_scores": [0.99],
                "rec_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }]

    monkeypatch.setattr(ocr_engine, "_get_paddle_ocr", lambda: FakePredictEngine())

    # Mock correct_orientation to return an upright image (height 300, width 100)
    upright = Image.new("RGB", (100, 300), color="white")
    monkeypatch.setattr(ocr_engine, "correct_orientation", lambda img: upright)

    sideways_img = Image.new("RGB", (300, 100), color="white")
    ocr_engine.read_paddle(sideways_img)

    # Confirm Paddle was passed the array from the upright image
    assert captured["shape"] == (300, 100, 3)


