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
    pass


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
                raise LLMError(f"Unknown provider: {self.provider}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM call failed [{self.provider}]: {str(e)}") from e

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

        # Gemini supports system instructions natively in newer SDK versions
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_prompt if system_prompt else None,
        )
        logger.info("[Gemini] Sending request to Google AI API...")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text.strip()

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
        response = client.chat.completions.create(
            model=settings.groq_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

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
        response = client.chat.completions.create(
            model=settings.openai_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

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
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()


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
