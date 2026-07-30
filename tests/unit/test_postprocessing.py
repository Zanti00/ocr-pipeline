import pytest

from app.core.postprocessing import DictionaryCorrector, LLMContextCorrector
from app.core.ocr_engine import OcrBundle, OcrReading, Word
from app.core.extraction import Extraction

@pytest.fixture
def mock_bundle():
    # Construct a mock OcrBundle
    words = [
        Word(text="Res", confidence=0.75, left=0, top=0, width=10, height=10, block=0, par=0, line=0),
        Word(text="Coffee", confidence=0.95, left=12, top=0, width=20, height=10, block=0, par=0, line=0),
        Word(text="Walmart", confidence=0.99, left=0, top=20, width=50, height=10, block=0, par=0, line=1),
        Word(text="Rxd", confidence=0.3, left=0, top=40, width=20, height=10, block=0, par=0, line=2),
        Word(text="Cf#e", confidence=0.4, left=22, top=40, width=20, height=10, block=0, par=0, line=2),
    ]
    primary = OcrReading(text="Res Coffee\nWalmart\nRxd Cf#e", words=words, confidence=0.7, variant="mock", psm=6)
    return OcrBundle(primary=primary)

def test_dictionary_corrector_moderate_confidence(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Res" should be corrected to "Red" since its confidence is 0.75 (between 0.5 and 0.9) and dist is 1.
    result = corrector.correct_word("Res", mock_bundle)
    assert result is not None
    assert result.corrected == "Red"
    
def test_dictionary_corrector_high_confidence_skip(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Walmart" should NOT be corrected (or even tried) if we assume it's correct. 
    # But wait, Walmart is in the vocab anyway. What if we misspelled it as Walmrt with 0.99 confidence?
    # Let's test a word not in vocab with high confidence.
    mock_bundle.primary.words.append(Word(text="Waalmart", confidence=0.95, left=0, top=0, width=10, height=10, block=0, par=0, line=0))
    result = corrector.correct_word("Waalmart", mock_bundle)
    assert result is None  # Skipped because confidence 0.95 > 0.90

def test_dictionary_corrector_low_confidence_skip(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Rxd" has confidence 0.3. It shouldn't be corrected.
    result = corrector.correct_word("Rxd", mock_bundle)
    assert result is None  # Skipped because confidence 0.3 < 0.50

def test_dictionary_corrector_phrase(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    corrected_phrase = corrector.correct_phrase("Res Coffee", mock_bundle)
    # Res -> Red, Coffee remains Coffee (either via vocab or confidence)
    assert corrected_phrase == "Red Coffee"

@pytest.mark.asyncio
async def test_llm_corrector_mocked(monkeypatch):
    from app.llm.base import LLMProvider
    
    class MockProvider(LLMProvider):
        async def extract_receipt_fields(self, ocr_text: str) -> dict:
            return {}
        async def normalize_text(self, text: str, context: str) -> str:
            if "Res Coffee" in text:
                return "Red Coffee"
            return text
            
    import app.core.postprocessing
    monkeypatch.setattr(app.core.postprocessing, "create_provider", lambda: MockProvider())
    
    corrector = LLMContextCorrector()
    items = [{"name": "Res Coffee", "quantity": 1, "price": 2.50}]
    corrected = await corrector.correct_line_items(items, "Res Coffee 2.50")
    
    assert len(corrected) == 1
    assert corrected[0]["name"] == "Red Coffee"

@pytest.mark.asyncio
async def test_normalize_extraction_item_mutation(mock_bundle, monkeypatch):
    from app.llm.base import LLMProvider
    from app.core.items import ExtractedItem, ItemScan
    
    class MockProvider(LLMProvider):
        async def extract_receipt_fields(self, ocr_text: str) -> dict:
            return {}
        async def normalize_text(self, text: str, context: str) -> str:
            if "Res Coffee" in text:
                return "Red Coffee"
            return text
            
    import app.core.postprocessing
    monkeypatch.setattr(app.core.postprocessing, "create_provider", lambda: MockProvider())
    
    # Mock DictionaryCorrector to just return the phrase so it doesn't mangle our test string
    class MockDictCorrector:
        def correct_phrase(self, phrase, bundle):
            return phrase
    monkeypatch.setattr(app.core.postprocessing, "DictionaryCorrector", MockDictCorrector)
    
    # Create an Extraction object with real ExtractedItem objects
    item = ExtractedItem(name="Res Coffee", quantity=1, price=2.50)
    item.descriptions.append("Extra shot") # Descriptions should be cleared
    scan = ItemScan(items=[item])
    
    extraction = Extraction()
    extraction.item_scan = scan
    
    # Run the normalization
    await app.core.postprocessing.normalize_extraction(extraction, mock_bundle)
    
    # Verify the item's name was mutated and descriptions cleared, so full_name resolves correctly
    assert extraction.item_scan.items[0].name == "Red Coffee"
    assert len(extraction.item_scan.items[0].descriptions) == 0
    assert extraction.item_scan.items[0].full_name == "Red Coffee"
