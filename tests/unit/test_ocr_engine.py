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


@pytest.mark.asyncio
async def test_read_pooled_can_select_tesseract_and_preserve_all_candidates(monkeypatch):
    paddle = reading(
        "Paddle total 10.00", 4.0, 0.99,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88, variant="contrast")
    alternate = reading("Other total 10.00", 5.5, 0.90, variant="flat", psm=4)

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    
    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract, alternate])
        
    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=3)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.all_readings == [paddle, tesseract, alternate]
    assert bundle.primary not in bundle.supporting
    assert len(bundle.supporting) == 2


@pytest.mark.asyncio
async def test_read_pooled_early_exits_when_fast_path_scores_high(monkeypatch):
    fast = reading("Fast total 10.00", 7.5, 0.90)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": fast)
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: None)

    async def mock_read_best(*args, **kwargs):
        raise AssertionError("full pool must not run on fast-path exit")

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object())

    assert bundle.early_exit is True
    assert bundle.primary is fast
    assert bundle.all_readings == [fast]


@pytest.mark.asyncio
async def test_read_pooled_escalates_when_fast_path_scores_low(monkeypatch):
    weak = reading("Weak noise", 2.0, 0.30)
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": weak)
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: reading(
        "Paddle total 10.00", 5.0, 0.95, engine="paddle", variant="source", psm=0))

    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract])

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.early_exit is False
    assert bundle.primary is tesseract
    assert len(bundle.all_readings) == 2


@pytest.mark.asyncio
async def test_read_pooled_falls_back_to_pool_when_fast_path_fails(monkeypatch):
    def fail_fast(image, lang="eng"):
        raise RuntimeError("preprocessing broke")

    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)
    monkeypatch.setattr(ocr_engine, "read_fast", fail_fast)
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: reading(
        "Paddle total 10.00", 5.0, 0.95, engine="paddle", variant="source", psm=0))

    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract])

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.early_exit is False
    assert bundle.primary is tesseract


def test_fast_path_eligible_threshold():
    assert ocr_engine._fast_path_eligible(reading("x", ocr_engine.FAST_PATH_ANCHOR_SCORE, 0.9))
    assert not ocr_engine._fast_path_eligible(
        reading("x", ocr_engine.FAST_PATH_ANCHOR_SCORE - 0.01, 0.9)
    )


@pytest.mark.asyncio
async def test_read_pooled_early_exits_via_fallback_when_fast_path_scores_low(monkeypatch):
    weak = reading("Weak noise", 2.0, 0.30)
    fallback = reading("Fallback total 10.00", 7.5, 0.90, variant="fast_alt", psm=11)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": weak)
    monkeypatch.setattr(
        ocr_engine, "read_fast_fallback", lambda image, lang="eng": fallback
    )
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: None)

    async def mock_read_best(*args, **kwargs):
        raise AssertionError("full pool must not run when fallback satisfies")

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object())

    assert bundle.early_exit is True
    assert bundle.primary is fallback
    assert bundle.all_readings == [fallback]


@pytest.mark.asyncio
async def test_read_pooled_escalates_when_fallback_also_scores_low(monkeypatch):
    weak = reading("Weak noise", 2.0, 0.30)
    weak_fallback = reading("Weak noise too", 1.5, 0.25, variant="fast_alt", psm=11)
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": weak)
    monkeypatch.setattr(
        ocr_engine, "read_fast_fallback", lambda image, lang="eng": weak_fallback
    )
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: reading(
        "Paddle total 10.00", 5.0, 0.95, engine="paddle", variant="source", psm=0))

    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract])

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.early_exit is False
    assert bundle.primary is tesseract
    assert len(bundle.all_readings) == 2


@pytest.mark.asyncio
async def test_read_pooled_escalates_when_fallback_fails(monkeypatch):
    weak = reading("Weak noise", 2.0, 0.30)

    def fail_fallback(image, lang="eng"):
        raise RuntimeError("flat rendering broke")

    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": weak)
    monkeypatch.setattr(ocr_engine, "read_fast_fallback", fail_fallback)
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: reading(
        "Paddle total 10.00", 5.0, 0.95, engine="paddle", variant="source", psm=0))

    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract])

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.early_exit is False
    assert bundle.primary is tesseract


@pytest.mark.asyncio
async def test_read_pooled_falls_back_to_tesseract_when_paddle_fails(monkeypatch):
    tesseract = reading("Tesseract total 10.00", 6.0, 0.88)

    def fail_paddle(image):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ocr_engine, "read_paddle", fail_paddle)
    
    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract])
        
    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=2)

    assert bundle.primary is tesseract
    assert bundle.primary.engine == "tesseract"
    assert bundle.supporting == []


@pytest.mark.asyncio
async def test_read_pooled_does_not_duplicate_identical_text(monkeypatch):
    paddle = reading(
        "Same total 10.00", 5.0, 0.95,
        engine="paddle", variant="source", psm=0,
    )
    tesseract = reading("Same total 10.00", 5.0, 0.90)
    alternate = reading("Different total 10.00", 4.0, 0.80, variant="flat")

    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: paddle)
    
    async def mock_read_best(*args, **kwargs):
        return (tesseract, [tesseract, alternate])
        
    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    bundle = await ocr_engine.read_pooled(object(), pool_size=3)

    assert bundle.combined_text.count("Same total 10.00") == 1
    assert alternate in bundle.supporting


@pytest.mark.asyncio
async def test_read_best_filters_variants_from_settings(monkeypatch):
    from app.config import settings

    built = [
        type("V", (), {"label": "raw", "image": None})(),
        type("V", (), {"label": "flat", "image": None})(),
        type("V", (), {"label": "clean", "image": None})(),
        type("V", (), {"label": "contrast", "image": None})(),
        type("V", (), {"label": "sauvola", "image": None})(),
    ]

    monkeypatch.setattr(ocr_engine, "build_variants", lambda image: built)
    monkeypatch.setattr(
        settings, "ocr_pool_variants", "raw,flat,clean",
        raising=False,
    )

    def read_variant(image, psm, lang="eng"):
        return reading("text", 1.0, 0.5)

    monkeypatch.setattr(ocr_engine, "read_variant", read_variant)

    _, candidates = await ocr_engine.read_best(object(), psms=(6,))

    labels = {c.variant for c in candidates}
    assert labels == {"raw", "flat", "clean"}


@pytest.mark.asyncio
async def test_read_best_keeps_all_variants_when_settings_say_all(monkeypatch):
    from app.config import settings

    built = [
        type("V", (), {"label": "raw", "image": None})(),
        type("V", (), {"label": "sauvola", "image": None})(),
    ]

    monkeypatch.setattr(ocr_engine, "build_variants", lambda image: built)
    monkeypatch.setattr(settings, "ocr_pool_variants", "all", raising=False)

    def read_variant(image, psm, lang="eng"):
        return reading("text", 1.0, 0.5)

    monkeypatch.setattr(ocr_engine, "read_variant", read_variant)

    _, candidates = await ocr_engine.read_best(object(), psms=(6,))

    assert {c.variant for c in candidates} == {"raw", "sauvola"}


@pytest.mark.asyncio
async def test_read_pooled_resolves_psms_from_settings(monkeypatch):
    from app.config import settings

    fast = reading("Fast total 10.00", 7.5, 0.90)
    monkeypatch.setattr(ocr_engine, "read_fast", lambda image, lang="eng": fast)
    monkeypatch.setattr(ocr_engine, "read_paddle", lambda image: reading(
        "Paddle total 10.00", 5.0, 0.95, engine="paddle", variant="source", psm=0))
    monkeypatch.setattr(settings, "ocr_pool_psms", "6,11", raising=False)

    captured = {}

    async def mock_read_best(*args, **kwargs):
        captured["psms"] = kwargs.get("psms")
        return (fast, [fast])

    monkeypatch.setattr(ocr_engine, "read_best", mock_read_best)

    await ocr_engine.read_pooled(object(), pool_size=2, fast_path=False)

    assert captured["psms"] == (6, 11)


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
