"""
Ollama utilities: check if Ollama is running, auto-pull models, and warm up.

Called during FastAPI startup when LLM_PROVIDER=ollama.
"""

import json
import logging
import requests
import time

logger = logging.getLogger(__name__)


def get_ollama_base_url(host: str = "localhost", port: int = 11434) -> str:
    return f"http://{host}:{port}"


def is_ollama_running(base_url: str) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_local_models(base_url: str) -> list:
    """Return list of model names already downloaded locally."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def is_model_available(base_url: str, model_name: str) -> bool:
    """Check if a specific model is already downloaded."""
    local = list_local_models(base_url)
    model_base = model_name.split(":")[0]
    for name in local:
        if name == model_name or name.split(":")[0] == model_base:
            return True
    return False


def pull_model(base_url: str, model_name: str) -> bool:
    """
    Pull (download) a model from the Ollama registry.
    Streams progress to the logger. Returns True on success.

    This is a blocking call. Large models (7B ~ 4 GB) can take several
    minutes on first download. Subsequent starts use the cached model instantly.
    """
    logger.info(f"Pulling Ollama model '{model_name}'... (may take a few minutes on first run)")
    try:
        with requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if any(k in status for k in ("pulling", "verifying", "success")):
                            logger.info(f"  [{model_name}] {status}")
                        if data.get("status") == "success":
                            logger.info(f"Model '{model_name}' pulled successfully.")
                            return True
                    except Exception:
                        pass
        return True
    except Exception as e:
        logger.error(f"Failed to pull model '{model_name}': {e}")
        return False


def warmup_model(base_url: str, model_name: str) -> None:
    """
    Send a tiny inference request to force Ollama to load model weights into RAM.

    Why this matters:
    Ollama loads model weights lazily - only when the first real request arrives.
    On a slow device this can take 10-60s. If a real request arrives while the
    model is still loading, Ollama throws:
        "Tried to use SessionInfo before it was initialized"

    By sending a dummy request at startup we pay the loading cost once, up front,
    so all subsequent requests hit the already-loaded model instantly.
    """
    logger.info(f"Warming up Ollama model '{model_name}' (loading weights into RAM)...")
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        if r.status_code == 200:
            logger.info(f"Model '{model_name}' is loaded and ready.")
        else:
            logger.warning(f"Warmup returned status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Could not warm up model '{model_name}': {e}")


def ensure_model_ready(
    model_name: str,
    ollama_host: str = "localhost",
    ollama_port: int = 11434,
) -> None:
    """
    Called at startup when LLM_PROVIDER=ollama.
    1. Verify Ollama is running
    2. Pull the model if not already downloaded
    3. Warm up the model (load weights into RAM)
    """
    base_url = get_ollama_base_url(ollama_host, ollama_port)

    if not is_ollama_running(base_url):
        logger.warning(
            f"Ollama does not appear to be running at {base_url}. "
            "Start it with: ollama serve"
        )
        return

    logger.info(f"Ollama is running at {base_url}")

    if not is_model_available(base_url, model_name):
        logger.info(f"Model '{model_name}' not found locally. Pulling now...")
        success = pull_model(base_url, model_name)
        if not success:
            logger.warning(
                f"Could not auto-pull '{model_name}'. "
                f"Pull it manually with: ollama pull {model_name}"
            )
            return
    else:
        logger.info(f"Model '{model_name}' is already downloaded.")

    # Load model weights into RAM now so the first real request is instant.
    # This prevents the "SessionInfo not initialized" race condition.
    warmup_model(base_url, model_name)
