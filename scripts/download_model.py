from sentence_transformers import SentenceTransformer

def download_model():
    # This will download and cache the model in the default Hugging Face cache directory
    print("Downloading all-MiniLM-L6-v2...")
    SentenceTransformer('all-MiniLM-L6-v2')
    print("Download complete.")

if __name__ == "__main__":
    download_model()
