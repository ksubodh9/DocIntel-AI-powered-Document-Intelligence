import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import FeedbackModal from "@/components/feedback/FeedbackModal";
import { submitFeedback, getEngagement } from "@/lib/api";

const FeedbackContext = createContext(null);

// Throttling knobs.
const COOLDOWN_AFTER_SUBMIT_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
const COOLDOWN_AFTER_SKIP_MS = 7 * 24 * 60 * 60 * 1000;    // 7 days
const COOLDOWN_KEY = "docintel_fb_until";
const SESSION_SHOWN_KEY = "docintel_fb_shown"; // sessionStorage — once per tab session

function readCooldownUntil() {
  try {
    return Number(localStorage.getItem(COOLDOWN_KEY)) || 0;
  } catch {
    return 0;
  }
}
function setCooldown(ms) {
  try {
    localStorage.setItem(COOLDOWN_KEY, String(Date.now() + ms));
  } catch {
    /* ignore */
  }
}
function wasShownThisSession() {
  try {
    return sessionStorage.getItem(SESSION_SHOWN_KEY) === "1";
  } catch {
    return false;
  }
}
function markShownThisSession() {
  try {
    sessionStorage.setItem(SESSION_SHOWN_KEY, "1");
  } catch {
    /* ignore */
  }
}

/**
 * Owns the feedback prompt lifecycle.
 *
 * eligible() gates every trigger: the user must have used a feature this
 * session, must not have been prompted already this session, and must be past
 * any cooldown window. requestFeedback() returns a Promise that resolves once
 * the prompt is resolved (sent or skipped) — or immediately if not eligible —
 * so callers like "log out" can await it and then continue.
 */
export function FeedbackProvider({ children }) {
  const [open, setOpen] = useState(false);
  const resolveRef = useRef(null);

  const eligible = useCallback(() => {
    if (!getEngagement()) return false;
    if (wasShownThisSession()) return false;
    if (Date.now() < readCooldownUntil()) return false;
    return true;
  }, []);

  const finish = useCallback(() => {
    setOpen(false);
    const r = resolveRef.current;
    resolveRef.current = null;
    if (r) r();
  }, []);

  const requestFeedback = useCallback(() => {
    if (open) return Promise.resolve();
    if (!eligible()) return Promise.resolve();
    markShownThisSession();
    setOpen(true);
    return new Promise((resolve) => {
      resolveRef.current = resolve;
    });
  }, [open, eligible]);

  const handleSubmit = useCallback(async ({ rating, comment }) => {
    try {
      await submitFeedback({
        rating,
        comment,
        route: typeof window !== "undefined" ? window.location.pathname : null,
        lastFeature: getEngagement(),
      });
    } catch {
      // Never block the user on a failed feedback POST.
    } finally {
      setCooldown(COOLDOWN_AFTER_SUBMIT_MS);
      finish();
    }
  }, [finish]);

  const handleSkip = useCallback(() => {
    setCooldown(COOLDOWN_AFTER_SKIP_MS);
    finish();
  }, [finish]);

  // Exit-intent: cursor leaving toward the top of the viewport (closing the tab
  // / switching away). Non-blocking — just surfaces the prompt if eligible.
  useEffect(() => {
    const onMouseOut = (e) => {
      if (e.clientY <= 0 && !e.relatedTarget) requestFeedback();
    };
    document.addEventListener("mouseout", onMouseOut);
    return () => document.removeEventListener("mouseout", onMouseOut);
  }, [requestFeedback]);

  return (
    <FeedbackContext.Provider value={{ requestFeedback }}>
      {children}
      <FeedbackModal open={open} onSubmit={handleSubmit} onSkip={handleSkip} />
    </FeedbackContext.Provider>
  );
}

export function useFeedback() {
  return useContext(FeedbackContext) || { requestFeedback: () => Promise.resolve() };
}
