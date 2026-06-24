import { useEffect, useState, useCallback } from "react";
import { Key, Server, ShieldCheck, Loader2, Check, Trash2, ArrowLeft, X, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { toast } from "@/components/ui/toaster";
import {
  getSessionMode, validateCredential, saveCredential, listCredentials, deleteCredential,
} from "@/lib/api";
import {
  getChoice, useServerDefault, useSessionKey, useSavedKey, clear as clearByok,
} from "@/lib/byok";

const PROVIDER_LABELS = {
  openai: "OpenAI", gemini: "Google Gemini", anthropic: "Anthropic (Claude)",
  groq: "Groq", huggingface: "HuggingFace", ollama: "Ollama (local)",
};
const label = (p) => PROVIDER_LABELS[p] || p;

// Used if /session/mode is unreachable (e.g. backend not yet restarted) so the
// provider dropdown is never empty.
const FALLBACK_PROVIDERS = ["openai", "gemini", "anthropic", "groq", "huggingface"];

// Providers a browser user can bring a key for (ollama is a server-local runtime).
const selectable = (providers) => (providers || []).filter((p) => p !== "ollama");

export default function ApiKeyDialog({ open, onClose }) {
  const [mode, setMode] = useState(null);            // /session/mode payload
  const [saved, setSaved] = useState([]);            // saved credentials list
  const [view, setView] = useState("choose");        // 'choose' | 'ownkey'
  const [loading, setLoading] = useState(false);

  // own-key form
  const [provider, setProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [persist, setPersist] = useState(false);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validated, setValidated] = useState(null);  // null | true | false

  const choice = getChoice();

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const m = await getSessionMode();
      setMode(m);
      const provs = selectable(m.supported_providers);
      const opts = provs.length ? provs : FALLBACK_PROVIDERS;
      if (!opts.includes(provider)) setProvider(opts[0]);
      if (m.persistence_available) {
        try {
          const list = await listCredentials();
          setSaved(list.credentials || []);
        } catch { setSaved([]); }
      } else {
        setSaved([]);
        setPersist(false);
      }
    } catch (e) {
      toast({ title: "Couldn't load key options", description: e.userMessage, variant: "error" });
    } finally {
      setLoading(false);
    }
  }, [provider]);

  useEffect(() => {
    if (open) { setView("choose"); setApiKey(""); setValidated(null); refresh(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const chooseDefault = () => {
    useServerDefault();
    toast({ title: "Using built-in keys", variant: "success" });
    onClose();
  };

  const onValidate = async () => {
    if (!apiKey) return;
    setValidating(true); setValidated(null);
    try {
      const r = await validateCredential({ provider, api_key: apiKey, model: model || undefined });
      setValidated(r.valid);
      toast({
        title: r.valid ? "Key is valid" : "Key check failed",
        description: r.message,
        variant: r.valid ? "success" : "error",
      });
    } catch (e) {
      setValidated(false);
      toast({ title: "Key check failed", description: e.userMessage, variant: "error" });
    } finally {
      setValidating(false);
    }
  };

  const onUseKey = async () => {
    if (!apiKey) {
      toast({ title: "Enter an API key first", variant: "error" });
      return;
    }
    if (persist) {
      setSaving(true);
      try {
        await saveCredential({ provider, api_key: apiKey, model: model || undefined, validate_first: true });
        useSavedKey({ provider, model: model || null });
        toast({ title: "Key saved", description: `${label(provider)} will be used automatically.`, variant: "success" });
        onClose();
      } catch (e) {
        toast({ title: "Key not saved", description: e.userMessage, variant: "error" });
      } finally {
        setSaving(false);
      }
    } else {
      // Session-only: keep client-side, no server storage.
      useSessionKey({ provider, apiKey, model: model || null });
      toast({ title: "Using your key for this session", variant: "success" });
      onClose();
    }
  };

  const onUseSaved = (cred) => {
    useSavedKey({ provider: cred.provider, model: cred.model });
    toast({ title: `Using your saved ${label(cred.provider)} key`, variant: "success" });
    onClose();
  };

  const onDeleteSaved = async (cred) => {
    try {
      await deleteCredential(cred.provider);
      // If the active choice was this saved key, fall back to default.
      if (choice?.mode === "saved" && choice.provider === cred.provider) clearByok();
      toast({ title: `Removed ${label(cred.provider)} key`, variant: "success" });
      refresh();
    } catch (e) {
      toast({ title: "Couldn't remove key", description: e.userMessage, variant: "error" });
    }
  };

  const activeLabel =
    choice?.mode === "default" ? "Built-in keys"
    : choice?.mode === "session" ? `Your ${label(choice.provider)} key (this session)`
    : choice?.mode === "saved" ? `Your saved ${label(choice.provider)} key`
    : null;

  const providerOptions = (() => {
    const fromMode = selectable(mode?.supported_providers);
    return fromMode.length ? fromMode : FALLBACK_PROVIDERS;
  })();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true">
      <div className="w-full max-w-md rounded-xl border bg-white shadow-xl dark:bg-slate-900 dark:border-slate-800">
        {/* header */}
        <div className="flex items-center justify-between border-b px-5 py-3 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold">AI provider &amp; keys</h2>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-md p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5">
          {activeLabel && (
            <p className="mb-3 text-xs text-muted-foreground">
              Currently using: <span className="font-medium text-foreground">{activeLabel}</span>
            </p>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          ) : view === "choose" ? (
            <div className="space-y-3">
              {mode?.server_default_available && (
                <button onClick={chooseDefault}
                  className="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:border-indigo-400 hover:bg-indigo-50/50 dark:border-slate-700 dark:hover:bg-slate-800">
                  <Server className="mt-0.5 h-5 w-5 text-slate-500" />
                  <div>
                    <p className="text-sm font-medium">Continue with built-in keys</p>
                    <p className="text-xs text-muted-foreground">Use the app's configured provider. Nothing to enter.</p>
                  </div>
                </button>
              )}

              <button onClick={() => setView("ownkey")}
                className="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:border-indigo-400 hover:bg-indigo-50/50 dark:border-slate-700 dark:hover:bg-slate-800">
                <Key className="mt-0.5 h-5 w-5 text-indigo-600 dark:text-indigo-400" />
                <div>
                  <p className="text-sm font-medium">Use my own API key</p>
                  <p className="text-xs text-muted-foreground">Run on your own provider quota — session-only or saved.</p>
                </div>
              </button>

              {saved.length > 0 && (
                <div className="pt-2">
                  <p className="mb-2 text-xs font-medium text-muted-foreground">Saved keys</p>
                  <div className="space-y-2">
                    {saved.map((c) => (
                      <div key={c.provider} className="flex items-center justify-between rounded-lg border p-2.5 dark:border-slate-700">
                        <div className="min-w-0">
                          <p className="flex items-center gap-2 text-sm font-medium">
                            {label(c.provider)}
                            {c.is_active && <Badge variant="green" className="text-[10px]">active</Badge>}
                          </p>
                          <p className="truncate text-xs text-muted-foreground">{c.masked_key}{c.model ? ` · ${c.model}` : ""}</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <Button size="sm" variant="ghost" onClick={() => onUseSaved(c)}>Use</Button>
                          <button onClick={() => onDeleteSaved(c)} aria-label="Delete key"
                            className="rounded-md p-1.5 text-slate-500 hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-950">
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <button onClick={() => setView("choose")} className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
                <ArrowLeft className="h-3.5 w-3.5" /> Back
              </button>

              <div>
                <label className="mb-1 block text-xs font-medium">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => { setProvider(e.target.value); setValidated(null); }}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  {providerOptions.map((p) => (
                    <option key={p} value={p}>{label(p)}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium">API key</label>
                <Input type="password" autoComplete="off" placeholder="Paste your key"
                  value={apiKey} onChange={(e) => { setApiKey(e.target.value); setValidated(null); }} />
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium">Model <span className="text-muted-foreground">(optional)</span></label>
                <Input placeholder="Leave blank for the provider default"
                  value={model} onChange={(e) => setModel(e.target.value)} />
              </div>

              {/* storage choice */}
              <div className="space-y-2 rounded-lg border p-3 dark:border-slate-700">
                <label className="flex items-start gap-2 cursor-pointer">
                  <input type="radio" name="persist" checked={!persist} onChange={() => setPersist(false)} className="mt-1" />
                  <span>
                    <span className="block text-sm font-medium">This session only</span>
                    <span className="block text-xs text-muted-foreground">Held in your browser, never stored on our server. Cleared when you close the tab.</span>
                  </span>
                </label>
                <label className={`flex items-start gap-2 ${mode?.persistence_available ? "cursor-pointer" : "opacity-50 cursor-not-allowed"}`}>
                  <input type="radio" name="persist" disabled={!mode?.persistence_available}
                    checked={persist} onChange={() => setPersist(true)} className="mt-1" />
                  <span>
                    <span className="flex items-center gap-1 text-sm font-medium"><Lock className="h-3 w-3" /> Save for next time</span>
                    <span className="block text-xs text-muted-foreground">
                      {mode?.persistence_available
                        ? "Encrypted and stored on our server so you don't re-enter it."
                        : "Unavailable — the server has no encryption key configured."}
                    </span>
                  </span>
                </label>
              </div>

              <div className="flex items-center gap-2 pt-1">
                <Button variant="outline" size="sm" onClick={onValidate} disabled={!apiKey || validating}>
                  {validating ? <Loader2 className="h-4 w-4 animate-spin" />
                    : validated === true ? <Check className="h-4 w-4 text-green-600" />
                    : <ShieldCheck className="h-4 w-4" />}
                  Test
                </Button>
                <Button size="sm" className="flex-1" onClick={onUseKey} disabled={!apiKey || saving}>
                  {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  {persist ? "Validate & save" : "Use this session"}
                </Button>
              </div>

              <p className="text-[11px] leading-snug text-muted-foreground">
                Your key is sent over HTTPS and used only to call the provider on your behalf.
                Session keys are never written to our database.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
