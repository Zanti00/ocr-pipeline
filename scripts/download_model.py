from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
import logging

# Suppress PaddleOCR verbose logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def download_model():
    # This will download and cache the model in the default Hugging Face cache directory
    print("Downloading all-MiniLM-L6-v2...")
    SentenceTransformer('all-MiniLM-L6-v2')
    print("Download complete.")

    print("Downloading PaddleOCR English models...")
    PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print("Download complete.")

if __name__ == "__main__":
    download_model()
