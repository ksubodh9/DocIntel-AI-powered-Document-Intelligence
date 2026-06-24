"""
LLM service - adapter layer over multiple providers.

Credentials & configuration
---------------------------
The service is driven by an ``LLMConfig`` value object (provider, keys, models,
fallback chain) rather than reading the global settings singleton directly. This
is what makes Bring-Your-Own-Key possible: a request can supply its own
``LLMConfig`` while the default still comes from the server's .env.

  * ``LLMConfig.from_settings(settings)`` — server defaults (current behaviour).
  * ``LLMConfig.for_byok(provider, api_key, model)`` — a single user-supplied
    provider/key, with no cross-provider fallback.

Fallback chain (server mode):
  Set LLM_FALLBACK_CHAIN=groq,ollama to automatically retry with the next
  provider when the primary fails due to rate limits, auth errors, or timeouts.
  For Gemini set GEMINI_MODELS=...; for Groq set GROQ_MODELS=... to try several
  models in order within the same provider.
"""

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# Providers the service knows how to call. Any user-supplied provider name is
# validated against this set before it can select a code path.
ALLOWED_PROVIDERS = frozenset(
    {"gemini", "ollama", "groq", "huggingface", "openai", "anthropic"}
)


# Shown to end users when anything goes wrong. Deliberately generic — it must
# never reveal provider names, model names, API keys, or .env configuration.
GENERIC_USER_MESSAGE = "Something went wrong while processing the document. Please try again."
BUSY_USER_MESSAGE = "The service is busy right now. Please try again in a moment."


# Patterns for common provider key formats. Used to redact secrets from any text
# that might reach the logs (e.g. raw SDK exception strings).
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{6,}"      # OpenAI / Anthropic
    r"|AIza[A-Za-z0-9_\-]{6,}"     # Google / Gemini
    r"|gsk_[A-Za-z0-9_\-]{6,}"     # Groq
    r"|hf_[A-Za-z0-9_\-]{6,})"     # HuggingFace
)


def _scrub(text) -> str:
    """Redact anything that looks like an API key before it is logged."""
    return _SECRET_RE.sub("***REDACTED***", str(text))


class LLMError(Exception):
    """
    LLM error carrying two messages:

      * ``message``      — detailed, technical text for server logs only.
      * ``user_message`` — safe, generic text returned to the client. It never
                           leaks internal details (provider, model, .env, keys).

    ``retry_after`` is an optional hint (seconds) for rate-limit responses.
    """
    def __init__(self, message: str, retry_after: int = 0, user_message: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.user_message = user_message or GENERIC_USER_MESSAGE


# -------------------------------------------------------------------------
# Configuration value object
# -------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Everything the service needs to make a call, independent of global settings.

    `api_keys`, `models`, and `model_lists` are keyed by provider name so the
    server-mode fallback chain can span providers; BYOK mode populates only the
    single provider the user supplied.
    """
    provider: str
    fallback_chain: list[str] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    model_lists: dict[str, list[str]] = field(default_factory=dict)
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    def key_for(self, provider: str) -> str:
        return self.api_keys.get(provider, "")

    def models_for(self, provider: str) -> list[str]:
        """Ordered model list to try for a provider (model_lists wins, else the single model)."""
        lst = self.model_lists.get(provider)
        if lst:
            return lst
        m = self.models.get(provider)
        return [m] if m else []

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_settings(cls, s) -> "LLMConfig":
        """Build the default config from the server's environment settings."""
        fallback = [p.strip().lower() for p in s.llm_fallback_chain.split(",") if p.strip()]
        model_lists = {}
        gem = [m.strip() for m in s.gemini_models.split(",") if m.strip()]
        grq = [m.strip() for m in s.groq_models.split(",") if m.strip()]
        if gem:
            model_lists["gemini"] = gem
        if grq:
            model_lists["groq"] = grq
        return cls(
            provider=s.llm_provider,
            fallback_chain=fallback,
            api_keys={
                "openai": s.openai_api_key,
                "gemini": s.gemini_api_key,
                "anthropic": s.anthropic_api_key,
                "groq": s.groq_api_key,
                "huggingface": s.huggingface_api_key,
            },
            models={
                "openai": s.openai_model,
                "gemini": s.gemini_model,
                "anthropic": s.anthropic_model,
                "groq": s.groq_model,
                "huggingface": s.huggingface_model,
                "ollama": s.ollama_model,
            },
            model_lists=model_lists,
            ollama_host=s.ollama_host,
            ollama_port=s.ollama_port,
        )

    @classmethod
    def for_byok(cls, provider: str, api_key: str, model: Optional[str] = None) -> "LLMConfig":
        """
        Build a config from a single user-supplied provider + key (BYOK).
        No cross-provider fallback: the user gave us one key, so we only use it.
        Falls back to the server's default model for the provider when none given.
        """
        provider = (provider or "").strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            raise LLMError(
                f"Unsupported provider '{provider}'.",
                user_message="That provider isn't supported.",
            )
        s = get_settings()
        default_model = model or {
            "openai": s.openai_model,
            "gemini": s.gemini_model,
            "anthropic": s.anthropic_model,
            "groq": s.groq_model,
            "huggingface": s.huggingface_model,
            "ollama": s.ollama_model,
        }.get(provider, "")
        return cls(
            provider=provider,
            fallback_chain=[],
            api_keys={provider: api_key},
            models={provider: default_model} if default_model else {},
            ollama_host=s.ollama_host,
            ollama_port=s.ollama_port,
        )


def _classify_api_error(provider: str, exc: Exception) -> "LLMError":
    """Convert a raw SDK exception into a clean LLMError (with secrets redacted)."""
    raw = _scrub(exc)
    lower = raw.lower()

    if any(k in raw for k in ("429", "RESOURCE_EXHAUSTED")) or \
       any(k in lower for k in ("quota", "rate limit", "rate_limit", "too many requests")):
        m = re.search(r"retry[_ ](?:in|delay)[^\d]*(\d+\.?\d*)", raw, re.IGNORECASE)
        wait = int(float(m.group(1))) + 1 if m else 0
        logger.debug(f"[LLM] Rate-limit from {provider}: {raw}")
        return LLMError(
            f"Rate limit reached ({provider}). retry_after={wait}s. {raw}",
            retry_after=wait,
            user_message=BUSY_USER_MESSAGE,
        )

    if any(k in raw for k in ("401", "403")) or \
       any(k in lower for k in ("api key", "api_key", "authentication", "unauthorized",
                                "permission denied", "invalid api key")):
        logger.debug(f"[LLM] Auth error from {provider}: {raw}")
        return LLMError(f"Auth error ({provider}). Check {provider.upper()}_API_KEY in .env. {raw}")

    if "404" in raw or ("not found" in lower and "model" in lower):
        logger.debug(f"[LLM] 404 from {provider}: {raw}")
        return LLMError(f"Model not found ({provider}). Check {provider.upper()}_MODEL in .env. {raw}")

    if any(k in raw for k in ("500", "502", "503", "504")) or \
       any(k in lower for k in ("service unavailable", "internal server error", "overloaded")):
        logger.debug(f"[LLM] Server error from {provider}: {raw}")
        return LLMError(
            f"Server error from {provider}: {raw}",
            user_message="The service is temporarily unavailable. Please try again shortly.",
        )

    if any(k in lower for k in ("timeout", "timed out", "read timeout", "connect timeout")):
        logger.debug(f"[LLM] Timeout from {provider}: {raw}")
        return LLMError(
            f"Timeout from {provider}: {raw}",
            user_message="The request took too long. Please try again.",
        )

    if any(k in lower for k in ("connection", "network", "unreachable", "failed to connect")):
        logger.debug(f"[LLM] Network error from {provider}: {raw}")
        return LLMError(
            f"Network error reaching {provider}: {raw}",
            user_message="The service is temporarily unavailable. Please try again shortly.",
        )

    logger.debug(f"[LLM] Unclassified error from {provider}: {raw}")
    return LLMError(f"Unclassified error from {provider}: {raw}")


def _is_fallback_worthy(err: LLMError) -> bool:
    """
    Return True if this error warrants trying the next provider/model.
    Rate limits, key errors, model-not-found, server errors all qualify.
    Invalid-request / content-filter errors do NOT (retrying won't help).
    """
    msg = str(err).lower()
    return any(k in msg for k in (
        "rate limit", "quota", "api key", "api_key", "authentication",
        "unauthorized", "model not found", "unavailable", "timed out",
        "cannot reach", "timeout", "overloaded", "bad gateway",
        "permission denied", "invalid api key", "server error", "network error",
    ))


class LLMService:
    def __init__(self, config: Optional[LLMConfig] = None):
        # Default to server settings; a caller (BYOK) may inject its own config.
        self.cfg = config or LLMConfig.from_settings(get_settings())
        self.provider = self.cfg.provider

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        chain = self._build_chain()
        last_error: Optional[LLMError] = None

        for provider in chain:
            try:
                result = self._call_provider(provider, prompt, system_prompt, temperature, max_tokens)
                if provider != self.provider:
                    logger.warning(f"[LLM] Fell back to {provider} (primary={self.provider})")
                return result
            except LLMError as e:
                last_error = e
                if _is_fallback_worthy(e) and len(chain) > 1:
                    logger.warning(f"[LLM] {provider} failed: {e} — trying next in chain")
                    continue
                raise

        raise last_error or LLMError("All providers in fallback chain failed.")

    def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        raw = self.complete(prompt, system_prompt, temperature=0.0)
        return extract_json(raw)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_chain(self) -> list[str]:
        """Return ordered list of provider names to try."""
        chain = [self.provider]
        for p in self.cfg.fallback_chain:
            if p and p not in chain:
                chain.append(p)
        return chain

    def _call_provider(
        self, provider: str, prompt: str, system_prompt: Optional[str],
        temperature: float, max_tokens: int,
    ) -> str:
        if provider not in ALLOWED_PROVIDERS:
            raise LLMError(f"Unknown provider '{provider}'. Check LLM_PROVIDER in .env.")

        logger.info(f"[LLM] Provider={provider}  temp={temperature}  max_tokens={max_tokens}")
        if system_prompt:
            logger.info(f"[LLM] System ({len(system_prompt)} chars): {system_prompt[:200]}")
        logger.info(f"[LLM] Prompt ({len(prompt)} chars): {prompt[:400]}")

        t0 = time.perf_counter()
        try:
            if provider == "gemini":
                result = self._gemini_complete(prompt, system_prompt, temperature, max_tokens)
            elif provider == "ollama":
                result = self._ollama_complete(prompt, system_prompt, temperature, max_tokens)
            elif provider == "groq":
                result = self._groq_complete(prompt, system_prompt, temperature, max_tokens)
            elif provider == "huggingface":
                result = self._huggingface_complete(prompt, system_prompt, temperature, max_tokens)
            elif provider == "openai":
                result = self._openai_complete(prompt, system_prompt, temperature, max_tokens)
            elif provider == "anthropic":
                result = self._anthropic_complete(prompt, system_prompt, temperature, max_tokens)
            else:  # pragma: no cover - guarded above
                raise LLMError(f"Unknown provider '{provider}'.")
        except LLMError:
            raise
        except Exception as e:
            raise _classify_api_error(provider, e) from e

        elapsed = time.perf_counter() - t0
        logger.info(f"[LLM] {provider} responded in {elapsed:.1f}s ({len(result)} chars): {result[:200]}")
        return result

    # -------------------------------------------------------------------------
    # Provider implementations
    # -------------------------------------------------------------------------

    def _gemini_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.cfg.key_for("gemini"))

        models = self.cfg.models_for("gemini")
        last_err: Optional[LLMError] = None
        for model_name in models:
            try:
                logger.info(f"[Gemini] Trying model: {model_name}")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt if system_prompt else None,
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return response.text.strip()
            except Exception as e:
                err = _classify_api_error("gemini", e)
                if err.retry_after > 0 or "rate limit" in str(err).lower():
                    last_err = err
                    logger.warning(f"[Gemini] {model_name} rate-limited, trying next model...")
                    continue
                raise err

        raise last_err or LLMError("All Gemini models exhausted.")

    def _ollama_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import ollama
        model = self.cfg.models.get("ollama") or "llama3.2:latest"
        logger.info(f"[Ollama] Model: {model}")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        current_messages = messages
        for attempt in range(3):
            try:
                response = ollama.chat(
                    model=model,
                    messages=current_messages,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                content = response.message.content
                if content is None:
                    raise LLMError("Ollama returned empty response.")
                return content.strip()
            except LLMError:
                raise
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "sessioninfo" in err_str or "not initialized" in err_str:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"[Ollama] Not ready, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if "bad message format" in err_str and system_prompt and attempt == 0:
                    logger.warning("[Ollama] Merging system prompt into user message...")
                    current_messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
                    continue
                raise _classify_api_error("ollama", e) from e
        raise LLMError(f"Ollama failed after 3 attempts: {last_error}")

    def _groq_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from groq import Groq

        models = self.cfg.models_for("groq")
        client = Groq(api_key=self.cfg.key_for("groq"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[LLMError] = None
        for model_name in models:
            try:
                logger.info(f"[Groq] Trying model: {model_name}")
                response = client.chat.completions.create(
                    model=model_name, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = _classify_api_error("groq", e)
                if err.retry_after > 0 or "rate limit" in str(err).lower():
                    last_err = err
                    logger.warning(f"[Groq] {model_name} rate-limited, trying next model...")
                    continue
                raise err

        raise last_err or LLMError("All Groq models exhausted.")

    def _huggingface_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from huggingface_hub import InferenceClient
        model = self.cfg.models.get("huggingface") or "mistralai/Mistral-7B-Instruct-v0.3"
        logger.info(f"[HuggingFace] Model: {model}")
        client = InferenceClient(model=model, token=self.cfg.key_for("huggingface"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat_completion(messages=messages, max_tokens=max_tokens,
                                          temperature=max(temperature, 0.01))
        return response.choices[0].message.content.strip()

    def _openai_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from openai import OpenAI
        model = self.cfg.models.get("openai") or "gpt-4o-mini"
        logger.info(f"[OpenAI] Model: {model}")
        client = OpenAI(api_key=self.cfg.key_for("openai"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise _classify_api_error("openai", e) from e

    def _anthropic_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import anthropic
        model = self.cfg.models.get("anthropic") or "claude-3-haiku-20240307"
        logger.info(f"[Anthropic] Model: {model}")
        client = anthropic.Anthropic(api_key=self.cfg.key_for("anthropic"))
        kwargs = dict(
            model=model, max_tokens=max_tokens,
            temperature=temperature, messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            raise _classify_api_error("anthropic", e) from e


# -------------------------------------------------------------------------
# JSON extraction
# -------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Could not parse JSON from LLM response:\n{text[:500]}")


def get_llm_service() -> LLMService:
    """
    FastAPI dependency: returns a service using the server's default config.

    BYOK request resolution is layered on top of this in the API layer (Step 3),
    which constructs an LLMService(LLMConfig.for_byok(...)) when a request carries
    its own credentials.
    """
    return LLMService()
