import axios from "axios";
import { supabase } from "./supabase";
import { getByokHeaders } from "./byok";

const _env = (typeof window !== "undefined" && window._env_) || {};
const BASE = _env.VITE_API_BASE || import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1";

const client = axios.create({ baseURL: BASE });

// Attach Supabase JWT to every request (best-effort — never blocks a request)
client.interceptors.request.use(async (config) => {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.access_token) {
      config.headers.Authorization = `Bearer ${session.access_token}`;
    }
  } catch {
    // No auth configured or session unavailable — proceed without token
  }
  // BYOK: attach per-request LLM credential headers based on the user's choice.
  try {
    Object.assign(config.headers, getByokHeaders());
  } catch {
    // Never let header resolution block a request.
  }
  return config;
});

// Normalize errors into { message, retry_after }
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail;
    if (typeof detail === "object" && detail?.message) {
      err.userMessage = detail.message;
      err.retryAfter = detail.retry_after || 0;
    } else if (typeof detail === "string") {
      err.userMessage = detail;
    } else {
      err.userMessage = err.message || "An unexpected error occurred.";
    }
    return Promise.reject(err);
  }
);

// ── Engagement tracking ─────────────────────────────────────────────────────
// Records that the user actually used a feature this session. The feedback
// prompt only fires for engaged users, so people who just bounce are never
// nagged. Stored in sessionStorage (cleared when the tab closes).
const ENGAGED_KEY = "docintel_engaged_feature";

export function markEngaged(feature) {
  try {
    sessionStorage.setItem(ENGAGED_KEY, feature || "1");
  } catch {
    /* storage unavailable (private mode) — non-fatal */
  }
}

export function getEngagement() {
  try {
    return sessionStorage.getItem(ENGAGED_KEY);
  } catch {
    return null;
  }
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function uploadDocument(file, onProgress) {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await client.post("/upload", fd, {
    timeout: 600_000,
    onUploadProgress: (e) => onProgress?.(Math.round((e.loaded / e.total) * 30)),
  });
  markEngaged("upload");
  return data;
}

export async function listDocuments() {
  const { data } = await client.get("/documents");
  return data;
}

export async function getDocument(docId) {
  const { data } = await client.get(`/document/${docId}`);
  return data;
}

export async function deleteDocument(docId) {
  await client.delete(`/document/${docId}`);
}

// ── Analysis ──────────────────────────────────────────────────────────────────

export async function classifyDocument(docId) {
  const { data } = await client.post(`/classify/${docId}`, {}, { timeout: 120_000 });
  markEngaged("classify");
  return data;
}

export async function summarizeDocument(docId) {
  const { data } = await client.post(`/summarize/${docId}`, {}, { timeout: 300_000 });
  markEngaged("summarize");
  return data;
}

export async function extractMetadata(docId) {
  const { data } = await client.post(`/metadata/${docId}`, {}, { timeout: 120_000 });
  markEngaged("metadata");
  return data;
}

export async function getTables(docId) {
  const { data } = await client.get(`/tables/${docId}`, { timeout: 60_000 });
  markEngaged("tables");
  return data;
}

export async function getSuggestedQuestions(docId) {
  const { data } = await client.get(`/questions/${docId}`, { timeout: 60_000 });
  return data;
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export async function chat(docId, message, includeHistory = true) {
  const { data } = await client.post("/chat", {
    document_id: docId, message, include_history: includeHistory,
  }, { timeout: 180_000 });
  markEngaged("chat");
  return data;
}

export async function multiChat(docIds, message) {
  const { data } = await client.post("/chat/multi", {
    document_ids: docIds, message,
  }, { timeout: 180_000 });
  markEngaged("chat_multi");
  return data;
}

export async function compareDocuments(docIdA, docIdB) {
  const { data } = await client.post("/compare", {
    document_id_a: docIdA, document_id_b: docIdB,
  }, { timeout: 240_000 });
  markEngaged("compare");
  return data;
}

// ── Feedback ──────────────────────────────────────────────────────────────────

export async function submitFeedback({ rating, comment, route, lastFeature }) {
  const { data } = await client.post("/feedback", {
    rating: rating ?? null,
    comment: comment || null,
    route: route || null,
    last_feature_used: lastFeature || null,
  }, { timeout: 20_000 });
  return data;
}

export async function getAdminFeedback() {
  const { data } = await client.get("/admin/feedback");
  return data;
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export async function getAdminStats() {
  const { data } = await client.get("/admin/stats");
  return data;
}

export async function getAdminUsers() {
  const { data } = await client.get("/admin/users");
  return data;
}

export async function getAdminUserDocuments(userId) {
  const { data } = await client.get(`/admin/users/${userId}/documents`);
  return data;
}

// ── BYOK credentials ──────────────────────────────────────────────────────────

export async function getSessionMode() {
  const { data } = await client.get("/session/mode");
  return data;
}

export async function validateCredential({ provider, api_key, model }) {
  const { data } = await client.post("/credentials/validate",
    { provider, api_key, model }, { timeout: 60_000 });
  return data;
}

export async function saveCredential({ provider, api_key, model, validate_first = true }) {
  const { data } = await client.post("/credentials",
    { provider, api_key, model, validate_first }, { timeout: 60_000 });
  return data;
}

export async function listCredentials() {
  const { data } = await client.get("/credentials");
  return data;
}

export async function deleteCredential(provider) {
  await client.delete(`/credentials/${provider}`);
}
