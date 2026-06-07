"""
LLM service - adapter layer over multiple providers.
Logs every step including the full prompt sent to the LLM.
"""

import json
import re
import time
import logging
from typing import Optional

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """
    User-facing LLM error with an optional retry_after hint (seconds).
    The message is always clean and human-readable — never a raw API dump.
    """
    def __init__(self, message: str, retry_after: int = 0):
        super().__init__(message)
        self.retry_after = retry_after  # 0 = no hint


def _classify_api_error(provider: str, exc: Exception) -> "LLMError":
    """
    Convert a raw SDK exception into a clean LLMError.
    Logs the full technical detail at DEBUG level; only returns a short message.
    """
    raw = str(exc)
    lower = raw.lower()

    # ── Rate limit / quota ────────────────────────────────────────────────────
    if any(k in raw for k in ("429", "RESOURCE_EXHAUSTED")) or \
       any(k in lower for k in ("quota", "rate limit", "rate_limit", "too many requests")):
        import re as _re
        m = _re.search(r"retry[_ ](?:in|delay)[^\d]*(\d+\.?\d*)", raw, _re.IGNORECASE)
        wait = int(float(m.group(1))) + 1 if m else 0
        hint = f" Please wait {wait}s before retrying." if wait else " Please wait a moment before retrying."
        logger.debug(f"[LLM] Raw rate-limit error from {provider}: {raw}")
        return LLMError(f"Rate limit reached ({provider}).{hint}", retry_after=wait)

    # ── Auth / API key ────────────────────────────────────────────────────────
    if any(k in raw for k in ("401", "403")) or \
       any(k in lower for k in ("api key", "api_key", "authentication", "unauthorized", "permission denied")):
        logger.debug(f"[LLM] Raw auth error from {provider}: {raw}")
        return LLMError(f"API key error ({provider}). Check {provider.upper()}_API_KEY in your .env file.")

    # ── Model not found ───────────────────────────────────────────────────────
    if "404" in raw or ("not found" in lower and "model" in lower):
        logger.debug(f"[LLM] Raw 404 error from {provider}: {raw}")
        return LLMError(
            f"Model not found ({provider}). Check {provider.upper()}_MODEL in your .env "
            f"(currently: {getattr(settings, provider + '_model', 'unknown')})."
        )

    # ── Server / service errors ───────────────────────────────────────────────
    if any(k in raw for k in ("500", "502", "503", "504")) or \
       any(k in lower for k in ("service unavailable", "internal server error", "bad gateway", "overloaded")):
        logger.debug(f"[LLM] Raw server error from {provider}: {raw}")
        return LLMError(f"The {provider} service is temporarily unavailable. Try again in a few seconds.")

    # ── Timeout / network ─────────────────────────────────────────────────────
    if any(k in lower for k in ("timeout", "timed out", "read timeout", "connect timeout")):
        logger.debug(f"[LLM] Raw timeout error from {provider}: {raw}")
        return LLMError(f"Request to {provider} timed out. The service may be overloaded — please try again.")

    if any(k in lower for k in ("connection", "network", "unreachable", "failed to connect")):
        logger.debug(f"[LLM] Raw network error from {provider}: {raw}")
        return LLMError(f"Cannot reach the {provider} API. Check your internet connection.")

    # ── Empty response ────────────────────────────────────────────────────────
    if any(k in lower for k in ("empty response", "no content", "none", "finish_reason")):
        logger.debug(f"[LLM] Empty response from {provider}: {raw}")
        return LLMError(f"The {provider} API returned an empty response. Please try again.")

    # ── Fallback — generic but clean ──────────────────────────────────────────
    logger.debug(f"[LLM] Unclassified error from {provider}: {raw}")
    return LLMError(f"The {provider} AI service returned an unexpected error. Please try again.")


class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        logger.info(f"[LLM] Provider={self.provider}  temp={temperature}  max_tokens={max_tokens}")
        if system_prompt:
            logger.info(f"[LLM] System prompt ({len(system_prompt)} chars): {system_prompt[:200]}...")
        logger.info(f"[LLM] User prompt ({len(prompt)} chars):\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")

        t0 = time.perf_counter()
        try:
            if self.provider == "gemini":
                result = self._gemini_complete(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "ollama":
                result = self._ollama_complete(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "groq":
                result = self._groq_complete(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "huggingface":
                result = self._huggingface_complete(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "openai":
                result = self._openai_complete(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "anthropic":
                result = self._anthropic_complete(prompt, system_prompt, temperature, max_tokens)
            else:
                raise LLMError(f"Unknown LLM provider '{self.provider}'. Check LLM_PROVIDER in your .env.")
        except LLMError:
            raise  # already classified and clean
        except Exception as e:
            # Catch anything the provider method didn't already wrap
            raise _classify_api_error(self.provider, e) from e

        elapsed = time.perf_counter() - t0
        logger.info(f"[LLM] Response received in {elapsed:.1f}s ({len(result)} chars): {result[:300]}{'...' if len(result) > 300 else ''}")
        return result

    def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        logger.info("[LLM] Expecting JSON response")
        raw = self.complete(prompt, system_prompt, temperature=0.0)
        result = extract_json(raw)
        logger.info(f"[LLM] Parsed JSON keys: {list(result.keys())}")
        return result

    # --------------------------------------------------------------------------
    # Gemini - free tier: 15 req/min, 1M tokens/day
    # Models: gemini-1.5-flash | gemini-1.5-pro | gemini-2.0-flash-exp
    # --------------------------------------------------------------------------

    def _gemini_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import google.generativeai as genai
        logger.info(f"[Gemini] Using model: {settings.gemini_model}")
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_prompt if system_prompt else None,
        )
        logger.info("[Gemini] Sending request to Google AI API...")
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return response.text.strip()
        except Exception as e:
            raise _classify_api_error("gemini", e) from e

    # --------------------------------------------------------------------------
    # Ollama - local, no API key needed
    # --------------------------------------------------------------------------

    def _ollama_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import ollama
        logger.info(f"[Ollama] Using model: {settings.ollama_model}")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        current_messages = messages
        for attempt in range(3):
            try:
                logger.info(f"[Ollama] Sending request (attempt {attempt+1}/3)...")
                response = ollama.chat(
                    model=settings.ollama_model,
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
                    logger.warning(f"[Ollama] Model not ready, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if "bad message format" in err_str and system_prompt and attempt == 0:
                    logger.warning("[Ollama] Merging system prompt into user message...")
                    current_messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
                    continue
                raise
        raise LLMError(f"Ollama failed after 3 attempts: {last_error}")

    # --------------------------------------------------------------------------
    # Groq - free tier
    # --------------------------------------------------------------------------

    def _groq_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from groq import Groq
        logger.info(f"[Groq] Using model: {settings.groq_model}")
        client = Groq(api_key=settings.groq_api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.info("[Groq] Sending request...")
        try:
            response = client.chat.completions.create(
                model=settings.groq_model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise _classify_api_error("groq", e) from e

    # --------------------------------------------------------------------------
    # HuggingFace
    # --------------------------------------------------------------------------

    def _huggingface_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from huggingface_hub import InferenceClient
        logger.info(f"[HuggingFace] Using model: {settings.huggingface_model}")
        client = InferenceClient(model=settings.huggingface_model, token=settings.huggingface_api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.info("[HuggingFace] Sending request...")
        response = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=max(temperature, 0.01))
        return response.choices[0].message.content.strip()

    # --------------------------------------------------------------------------
    # OpenAI
    # --------------------------------------------------------------------------

    def _openai_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        from openai import OpenAI
        logger.info(f"[OpenAI] Using model: {settings.openai_model}")
        client = OpenAI(api_key=settings.openai_api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.info("[OpenAI] Sending request...")
        try:
            response = client.chat.completions.create(
                model=settings.openai_model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise _classify_api_error("openai", e) from e

    # --------------------------------------------------------------------------
    # Anthropic
    # --------------------------------------------------------------------------

    def _anthropic_complete(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import anthropic
        logger.info(f"[Anthropic] Using model: {settings.anthropic_model}")
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        kwargs = dict(
            model=settings.anthropic_model, max_tokens=max_tokens,
            temperature=temperature, messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        logger.info("[Anthropic] Sending request...")
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            raise _classify_api_error("anthropic", e) from e


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

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
    return LLMService()
