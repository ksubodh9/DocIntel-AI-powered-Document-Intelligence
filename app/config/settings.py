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
    app_name: str = "Document Intelligence System"
    app_version: str = "2.0.0"
    debug: bool = False

    # LLM Provider — primary provider used for all calls
    # Options: ollama | groq | huggingface | openai | anthropic | gemini
    llm_provider: str = Field(default="gemini", description="Primary LLM provider")

    # Fallback chain — comma-separated providers tried in order when primary fails
    # Example: "gemini,groq" tries Gemini first, falls back to Groq on rate-limit/auth error
    llm_fallback_chain: str = Field(default="", description="Comma-separated fallback providers")

    # API keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key (free at console.groq.com)")
    huggingface_api_key: str = Field(default="", description="HuggingFace token")

    # LLM Models
    openai_model: str = "gpt-4o-mini"
    # Comma-separated Gemini models tried in order (rate-limit fallback within Gemini)
    # Valid names: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    gemini_model: str = "gemini-2.0-flash"
    gemini_models: str = Field(default="", description="Comma-separated Gemini model names to try in order")
    anthropic_model: str = "claude-3-haiku-20240307"
    groq_model: str = "llama-3.1-8b-instant"
    # Multiple Groq models tried in order on rate-limit
    groq_models: str = Field(default="", description="Comma-separated Groq model names to try in order")
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # Ollama local models
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
    allowed_extensions: list[str] = [
        ".pdf",
        ".docx", ".pptx", ".xlsx", ".xls",
        ".csv", ".txt", ".md",
        ".html", ".htm",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif",
    ]

    # RAG / Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    # Minimum cosine similarity score [0–1] a chunk must reach to be used as context
    # or returned as a citation.  Chunks below this are discarded before the LLM call.
    # BAAI/bge-small-en-v1.5 typical ranges:
    #   ≥ 0.5  clearly relevant   0.35–0.5  possibly relevant   < 0.35  likely noise
    min_relevance_score: float = 0.35

    # Re-ranking (retrieve-then-rerank). When enabled, the vector store over-fetches
    # `rerank_candidates` chunks, a cross-encoder reorders them, and the top
    # `top_k_retrieval` are kept. Disabled by default — adds ~50–200ms/query.
    # The cosine `min_relevance_score` filter still applies to each returned chunk
    # (rerank scores are unbounded logits, not cosine, so the same threshold can't
    # apply to them). The reranker only changes ordering, not which chunks pass.
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 20  # chunks pulled from ChromaDB before reranking

    # Rate Limiting
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  # seconds

    # Security
    # When True (dev only) JWTs are decoded WITHOUT signature verification if the
    # secret is missing. MUST stay False in production.
    allow_insecure_auth: bool = False
    # Max characters accepted for a chat question (limits prompt-injection surface
    # and token abuse).
    max_question_length: int = 4000
    # CORS: comma-separated exact origins allowed (in addition to localhost dev).
    cors_allowed_origins: str = Field(default="", description="Comma-separated allowed origins")
    # Optional regex for CORS origins (e.g. preview deploys). Empty = disabled.
    cors_allowed_origin_regex: str = Field(default="", description="Regex of allowed origins")
    # Comma-separated trusted Host headers. Empty = middleware disabled (allow all).
    allowed_hosts: str = Field(default="", description="Comma-separated trusted hosts")
    # Max uncompressed:compressed ratio allowed for zip-based Office files (bomb guard).
    max_decompression_ratio: int = 120

    # Supabase (required for auth in production)
    # JWT secret: Supabase dashboard -> Settings -> API -> JWT Secret
    supabase_jwt_secret: str = Field(default="", description="Supabase JWT secret for token verification")
    # Service role key: used by admin endpoints to query auth.users
    # Supabase dashboard -> Settings -> API -> service_role (secret)
    supabase_service_role_key: str = Field(default="", description="Supabase service role key (admin use only)")
    supabase_url: str = Field(default="", description="https://your-project.supabase.co")

    # BYOK persisted keys: Fernet key (urlsafe base64, 32 bytes) used to encrypt
    # saved provider API keys at rest. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If empty, SESSION-ONLY BYOK still works but SAVING keys is disabled.
    credentials_encryption_key: str = Field(default="", description="Fernet key for encrypting saved API keys")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton (reads .env once on first call)."""
    return Settings()
  