import { useState } from "react";
import { Star, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";

const MAX_COMMENT = 2000;

/**
 * Minimal two-field feedback form: a 1–5 star rating and an optional comment.
 * Kept deliberately small so it's a few-seconds ask. `onSubmit` receives
 * { rating, comment }; `onSkip` dismisses without sending.
 */
export default function FeedbackModal({ open, onSubmit, onSkip }) {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!open) return null;

  const canSend = rating > 0 || comment.trim().length > 0;

  const handleSend = async () => {
    if (!canSend || submitting) return;
    setSubmitting(true);
    try {
      await onSubmit({ rating: rating || null, comment: comment.trim() || null });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
        onClick={() => !submitting && onSkip()}
        aria-hidden="true"
      />

      {/* Card */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Share feedback"
        className="animate-fade-in relative w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-6 shadow-xl dark:border-slate-800 dark:bg-slate-900"
      >
        <button
          onClick={() => !submitting && onSkip()}
          className="absolute right-3 top-3 rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>

        <h2 className="text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          How was your experience?
        </h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Takes a few seconds — totally optional.
        </p>

        {/* Rating */}
        <div className="mt-5 flex items-center gap-1.5">
          {[1, 2, 3, 4, 5].map((n) => {
            const active = n <= (hover || rating);
            return (
              <button
                key={n}
                type="button"
                onClick={() => setRating(n === rating ? 0 : n)}
                onMouseEnter={() => setHover(n)}
                onMouseLeave={() => setHover(0)}
                className="rounded p-0.5 transition-transform hover:scale-110"
                aria-label={`${n} star${n > 1 ? "s" : ""}`}
              >
                <Star
                  className={
                    active
                      ? "h-7 w-7 fill-amber-400 text-amber-400"
                      : "h-7 w-7 text-slate-300 dark:text-slate-600"
                  }
                />
              </button>
            );
          })}
        </div>

        {/* Comment */}
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value.slice(0, MAX_COMMENT))}
          rows={3}
          placeholder="Comments or suggestions (optional)"
          className="mt-4 w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:ring-indigo-900/40"
        />

        {/* Actions */}
        <div className="mt-5 flex items-center justify-end gap-2">
          <Button variant="ghost" onClick={onSkip} disabled={submitting}>
            No thanks
          </Button>
          <Button onClick={handleSend} disabled={!canSend || submitting}>
            {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
            Send feedback
          </Button>
        </div>
      </div>
    </div>
  );
}
