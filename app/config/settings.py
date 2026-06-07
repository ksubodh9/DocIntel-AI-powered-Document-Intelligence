"""
Application settings loaded from environment variables.
Using pydantic-settings for type-safe config with .env support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # App
    app_name: str = "PDF Agent"
    app_version: str = "1.0.0"
    debug: bool = False

    # LLM Provider
    # Options: ollama | groq | huggingface | openai | anthropic | gemini
    llm_provider: str = Field(default="gemini", description="LLM provider to use")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key (free at console.groq.com)")
    huggingface_api_key: str = Field(default="", description="HuggingFace token")

    # LLM Models
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-flash-latest"  # stable alias, always points to latest Flash
    anthropic_model: str = "claude-3-haiku-20240307"
    groq_model: str = "llama-3.1-8b-instant"
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # Ollama local models - auto-pulled on startup if not present
    # Options: llama3.1:8b | mistral:7b | phi3:mini | gemma2:9b | llama3.2:3b
    ollama_model: str = "llama3.2:latest"
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"  # cpu | cuda

    # Storage
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    database_url: str = f"sqlite:///{BASE_DIR.as_posix()}/data/pdf_agent.db"

    # File Limits
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf"]

    # RAG / Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5

    # Rate Limiting
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_active_api_key(self) -> str:
        """Return the API key for the active LLM provider."""
        keys = {
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "huggingface": self.huggingface_api_key,
            "ollama": "",  # no key needed
        }
        return keys.get(self.llm_provider, "")


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton - import this everywhere."""
    return Settings()
