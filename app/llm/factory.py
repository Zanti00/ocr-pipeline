from app.llm.base import LLMProvider
from app.config import settings

def create_provider() -> LLMProvider:
    provider_name = settings.llm_provider.lower()
    
    if provider_name == "ollama":
        from app.llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    
    raise ValueError(f"Unsupported LLM provider: {provider_name}")
