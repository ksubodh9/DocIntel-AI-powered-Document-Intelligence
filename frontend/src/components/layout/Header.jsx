import { useEffect, useState } from "react";
import { FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const API_BASE = (window._env_?.VITE_API_BASE || import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1")
  .replace(/\/api\/v1\/?$/, "");

function BackendStatus() {
  // "checking" | "ok" | "error"
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(8000) });
        if (!cancelled) setStatus(res.ok ? "ok" : "error");
      } catch {
        if (!cancelled) setStatus("error");
      }
    };
    check();
    // Re-check every 60 s
    const id = setInterval(check, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const label  = { checking: "Connecting…", ok: "Backend connected", error: "Backend offline" }[status];
  const dot    = { checking: "bg-amber-400 animate-pulse", ok: "bg-green-500", error: "bg-red-500" }[status];
  const text   = { checking: "text-amber-600", ok: "text-green-700", error: "text-red-600" }[status];

  return (
    <div className={`flex items-center gap-1.5 text-xs font-medium ${text}`}>
      <span className={`h-2 w-2 rounded-full shrink-0 ${dot}`} />
      <span className="hidden sm:inline">{label}</span>
      {status === "error" && (
        <a
          href={`${API_BASE}/docs`}
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2 opacity-70 hover:opacity-100"
        >
          wake
        </a>
      )}
    </div>
  );
}

export default function Header({ document }) {
  if (!document) {
    return (
      <div className="flex h-14 items-center justify-between border-b bg-white px-6">
        <p className="text-sm text-muted-foreground">Upload a document to get started — PDF, Word, Excel, PowerPoint, CSV, HTML, or image</p>
        <BackendStatus />
      </div>
    );
  }

  const name = document.original_filename || document.filename || "Document";
  const docType = document.document_type;

  const typeColors = {
    "Research Paper": "indigo",
    "Legal Document": "rose",
    "Medical Report": "green",
    "Invoice": "amber",
    "Financial Report": "amber",
    "Resume": "violet",
  };

  return (
    <div className="flex h-14 items-center gap-3 border-b bg-white px-6">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 shrink-0">
        <FileText className="h-4 w-4 text-indigo-600" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-semibold text-slate-900">{name}</p>
          {docType && (
            <Badge variant={typeColors[docType] || "secondary"} className="hidden sm:inline-flex text-[10px]">
              {docType}
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {document.page_count ?? "?"} pages
          {document.status && (
            <span className={`ml-2 ${document.status === "ready" ? "text-green-600" : "text-amber-600"}`}>
              {document.status}
            </span>
          )}
        </p>
      </div>
      <BackendStatus />
    </div>
  );
}
