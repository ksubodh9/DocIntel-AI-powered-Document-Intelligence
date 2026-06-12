import { useState } from "react";
import { Loader2, Sparkles, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { classifyDocument, summarizeDocument, extractMetadata, getSuggestedQuestions } from "@/lib/api";
import { toast } from "@/components/ui/toaster";

function Chips({ items, color }) {
  if (!items?.length) return <span className="text-xs text-slate-400 italic">None found</span>;
  return (
    <div className="flex flex-wrap gap-1.5 mt-1">
      {items.map((item) => (
        <Badge key={item} variant={color} className="text-xs">{item}</Badge>
      ))}
    </div>
  );
}

function MetaGrid({ metadata }) {
  const LABELS = {
    title: "Title", author: "Author", date: "Date", language: "Language",
    tone: "Tone", target_audience: "Audience", reading_time_minutes: "Reading time",
    word_count: "Words", document_length: "Length",
    pdf_title: "PDF Title", pdf_author: "PDF Author", pdf_creation_date: "Created",
  };
  const items = Object.entries(metadata || {})
    .filter(([, v]) => v && v !== "null")
    .map(([k, v]) => [LABELS[k] || k.replace(/_/g, " "), v]);

  if (!items.length) return null;
  return (
    <div className="grid grid-cols-2 gap-2 mt-2">
      {items.map(([key, val]) => (
        <div key={key} className="rounded-lg bg-slate-50 border px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">{key}</p>
          <p className="text-sm font-semibold text-slate-800 truncate">{String(val)}</p>
        </div>
      ))}
    </div>
  );
}

export default function InsightsPanel({ document, onAskQuestion, onAnalysisComplete }) {
  const docId = document?.id;

  const [state, setState] = useState({ step: "idle", progress: 0, error: null });
  const [results, setResults] = useState({
    classify: document?.document_type
      ? { document_type: document.document_type, confidence: document.classification_confidence }
      : null,
    summary: document?.short_summary
      ? { short_summary: document.short_summary, detailed_summary: document.detailed_summary, topics: document.topics, keywords: document.keywords, entities: document.entities }
      : null,
    metadata: document?.doc_metadata || null,
    questions: document?.suggested_questions ? { questions: document.suggested_questions } : null,
  });
  const [showDetail, setShowDetail] = useState(false);

  const hasSomething = results.classify || results.summary;
  const isRunning = state.step === "running";

  const run = async () => {
    setState({ step: "running", progress: 10, error: null });
    try {
      setState((s) => ({ ...s, progress: 20 }));
      const classify = await classifyDocument(docId);
      setResults((r) => ({ ...r, classify }));

      setState((s) => ({ ...s, progress: 40 }));
      const summary = await summarizeDocument(docId);
      setResults((r) => ({ ...r, summary }));

      setState((s) => ({ ...s, progress: 65 }));
      const metaRes = await extractMetadata(docId);
      setResults((r) => ({ ...r, metadata: metaRes.metadata }));

      setState((s) => ({ ...s, progress: 85 }));
      const questions = await getSuggestedQuestions(docId);
      setResults((r) => ({ ...r, questions }));

      setState({ step: "done", progress: 100, error: null });
      // Notify parent to refresh doc data so results persist across tab switches
      onAnalysisComplete?.();
    } catch (err) {
      const msg = err.userMessage || "Something went wrong while analysing the document. Please try again.";
      setState({ step: "error", progress: 0, error: msg });
      toast({ title: "Analysis failed", description: msg, variant: "error" });
    }
  };

  return (
    <div className="flex flex-col gap-4 p-5">

      {/* ── Action bar — ALWAYS visible ─────────────────────────────────── */}
      <Card>
        <CardContent className="pt-4 pb-4 flex flex-col sm:flex-row items-center gap-3">
          {hasSomething && !isRunning && (
            <p className="text-xs text-slate-400 flex-1 hidden sm:block">
              Previous results shown · click to re-run
            </p>
          )}
          <Button
            onClick={run}
            disabled={isRunning}
            className="flex-1 sm:flex-none sm:min-w-[150px]"
          >
            {isRunning ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {hasSomething ? "Re-analyse" : "Analyse document"}
          </Button>
        </CardContent>
      </Card>

      {/* ── Progress ─────────────────────────────────────────────────────── */}
      {isRunning && (
        <Card>
          <CardContent className="pt-6 flex flex-col items-center gap-3 py-8">
            <Loader2 className="h-8 w-8 text-indigo-600 animate-spin" />
            <p className="text-sm text-slate-600">Analysing document…</p>
            <Progress value={state.progress} className="w-full max-w-xs" />
          </CardContent>
        </Card>
      )}

      {/* ── Error ────────────────────────────────────────────────────────── */}
      {state.step === "error" && (
        <div className="flex items-start gap-3 rounded-lg bg-red-50 border border-red-200 px-4 py-3">
          <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-red-800">Analysis failed</p>
            <p className="text-xs text-red-600 mt-0.5 break-words">{state.error}</p>
          </div>
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────────────────────── */}
      {!hasSomething && !isRunning && state.step !== "error" && (
        <div className="text-center py-10 text-slate-400 text-sm">
          Click <strong>Analyse document</strong> to classify, summarise, and extract topics from this document.
        </div>
      )}

      {/* ── Classification ───────────────────────────────────────────────── */}
      {results.classify && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Document type</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <p className="text-xl font-bold text-slate-900">{results.classify.document_type}</p>
              <div className="flex-1">
                <p className="text-xs text-slate-500 mb-1">Confidence</p>
                <Progress value={(results.classify.confidence || 0) * 100} />
                <p className="text-xs font-medium text-slate-600 mt-0.5">
                  {((results.classify.confidence || 0) * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Summary ──────────────────────────────────────────────────────── */}
      {results.summary && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Summary</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <p className="text-sm text-slate-700 leading-relaxed">{results.summary.short_summary}</p>
            {results.summary.detailed_summary && (
              <div>
                <button
                  onClick={() => setShowDetail((v) => !v)}
                  className="flex items-center gap-1 text-xs text-indigo-600 font-medium hover:text-indigo-500"
                >
                  {showDetail ? "▲ Hide" : "▼ Show"} detailed analysis
                </button>
                {showDetail && (
                  <p className="mt-2 text-sm text-slate-600 leading-relaxed whitespace-pre-line">
                    {results.summary.detailed_summary}
                  </p>
                )}
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium mb-1">Topics</p>
                <Chips items={results.summary.topics} color="indigo" />
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium mb-1">Keywords</p>
                <Chips items={results.summary.keywords} color="green" />
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium mb-1">Entities</p>
                <Chips items={results.summary.entities} color="amber" />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Metadata ─────────────────────────────────────────────────────── */}
      {results.metadata && Object.keys(results.metadata).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Metadata</CardTitle>
          </CardHeader>
          <CardContent>
            <MetaGrid metadata={results.metadata} />
          </CardContent>
        </Card>
      )}

      {/* ── Suggested questions ───────────────────────────────────────────── */}
      {results.questions?.questions?.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Suggested questions</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {results.questions.questions.map((q, i) => (
              <button
                key={i}
                onClick={() => onAskQuestion?.(q)}
                className="w-full text-left rounded-lg border border-slate-200 px-3 py-2.5 text-sm text-slate-700 hover:bg-indigo-50 hover:border-indigo-300 hover:text-indigo-900 transition-colors"
              >
                {q}
              </button>
            ))}
          </CardContent>
        </Card>
      )}

    </div>
  );
}
