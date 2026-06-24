// Bring-Your-Own-Key client store.
//
// Holds the user's per-session choice of which LLM credentials to use and turns
// it into the X-LLM-* headers the backend resolver understands:
//   - "default" → X-LLM-Use-Default: true        (use the server's keys)
//   - "session" → X-LLM-Provider / X-LLM-Api-Key  (key held client-side only)
//   - "saved"   → no headers                      (server uses the user's saved key)
//
// The choice and a session-only key live in sessionStorage, so they survive a
// page reload but are cleared when the tab closes. The key is never sent to our
// own storage backend in "session" mode and never written to localStorage.

const CHOICE_KEY = "byok.choice";       // { mode, provider, model }
const SESSION_KEY = "byok.sessionKey";  // raw key, session mode only
const EVENT = "byok:changed";

function _read(key) {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function _emit() {
  try {
    window.dispatchEvent(new Event(EVENT));
  } catch {
    /* SSR / no window — ignore */
  }
}

export function getChoice() {
  // { mode: 'default'|'session'|'saved', provider?, model? } or null if unset.
  return _read(CHOICE_KEY);
}

export function hasChosen() {
  return getChoice() !== null;
}

export function useServerDefault() {
  try {
    sessionStorage.setItem(CHOICE_KEY, JSON.stringify({ mode: "default" }));
    sessionStorage.removeItem(SESSION_KEY);
  } catch { /* ignore */ }
  _emit();
}

export function useSessionKey({ provider, apiKey, model }) {
  try {
    sessionStorage.setItem(CHOICE_KEY, JSON.stringify({ mode: "session", provider, model: model || null }));
    sessionStorage.setItem(SESSION_KEY, apiKey || "");
  } catch { /* ignore */ }
  _emit();
}

export function useSavedKey({ provider, model }) {
  // The key itself stays encrypted on the server; we only remember the choice.
  try {
    sessionStorage.setItem(CHOICE_KEY, JSON.stringify({ mode: "saved", provider, model: model || null }));
    sessionStorage.removeItem(SESSION_KEY);
  } catch { /* ignore */ }
  _emit();
}

export function clear() {
  try {
    sessionStorage.removeItem(CHOICE_KEY);
    sessionStorage.removeItem(SESSION_KEY);
  } catch { /* ignore */ }
  _emit();
}

// Headers to merge onto every API request based on the current choice.
export function getByokHeaders() {
  const choice = getChoice();
  if (!choice) return {};
  if (choice.mode === "default") {
    return { "X-LLM-Use-Default": "true" };
  }
  if (choice.mode === "session") {
    let key = "";
    try { key = sessionStorage.getItem(SESSION_KEY) || ""; } catch { /* ignore */ }
    if (!choice.provider || !key) return {};
    const h = { "X-LLM-Provider": choice.provider, "X-LLM-Api-Key": key };
    if (choice.model) h["X-LLM-Model"] = choice.model;
    return h;
  }
  // "saved" → no headers; the server resolves the user's active saved key.
  return {};
}

// Subscribe to changes (returns an unsubscribe fn). Handy for header chips/badges.
export function onChange(cb) {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}
