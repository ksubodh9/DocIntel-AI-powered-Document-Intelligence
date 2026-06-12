import { useState } from "react";
import { GitCompare, Loader2, CheckCircle2, ArrowLeftRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { compareDocuments } from "@/lib/api";
import { toast } from "@/components/ui/toaster";

export default function ComparePanel({ documents = [] }) {
  const [docA, setDocA] = useState("");
  const [docB, setDocB] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showDetail, setShowDetail] = useState(false);

  const docOptions = documents.map((d) => ({
    id: d.id,
    name: (d.original_filename || d.filename || "Untitled").replace(/\.[^.]+$/, ""),
  }));

  const run = async () => {
    if (!docA || !docB || docA === docB) {
      toast({ title: "Select two different documents", variant: "error" });
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await compareDocuments(docA, docB);
      setResult(res);
    } catch (err) {
      toast({ title: "Comparison failed", description: err.userMessage, variant: "error" });
    } finally {
      setLoading(false);
    }
  };

  if (documents.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <GitCompare className="h-10 w-10 text-slate-300" />
        <p className="text-sm text-slate-500">Upload at least two documents to compare them.</p>
      </div>
    );
  }

  const nameA = docOptions.find((d) => d.id === docA)?.name;
  const nameB = docOptions.find((d) => d.id === docB)?.name;

  return (
    <div className="flex flex-col gap-4 p-5 overflow-y-auto">
      {/* Selectors */}
      <Card>
        <CardContent className="pt-5">
          <div className="flex flex-col sm:flex-row items-center gap-3">
            <div className="flex-1 w-full">
              <p className="text-xs text-slate-500 mb-1 font-medium">Document A</p>
              <select
                value={docA}
                onChange={(e) => setDocA(e.target.value)}
                className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
              >
                <option value="">Select document...</option>
                {docOptions.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
            </div>
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-100 shrink-0 mt-4">
              <ArrowLeftRight className="h-4 w-4 text-slate-500" />
            </div>
            <div className="flex-1 w-full">
              <p className="text-xs text-slate-500 mb-1 font-medium">Document B</p>
              <select
                value={docB}
                onChange={(e) => setDocB(e.target.value)}
                className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
              >
                <option value="">Select document...</option>
                {docOptions.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
            </div>
          </div>
          <Button
            onClick={run}
            disabled={loading || !docA || !docB}
            className="mt-4 w-full"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? "Comparing..." : "Compare documents"}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Similarities */}
            <Card className="border-green-200">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-green-700">Similarities</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="flex flex-col gap-1.5">
                  {result.similarities.map((s, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-green-500" />
                      {s}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Differences */}
            <Card className="border-orange-200">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-orange-700">Differences</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="flex flex-col gap-1.5">
                  {result.differences.map((d, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-orange-400" />
                      {d}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Recommendation */}
          <Card className="border-indigo-200 bg-indigo-50">
            <CardContent className="pt-5">
              <p className="text-xs uppercase tracking-wider text-indigo-500 font-semibold mb-1.5">Recommendation</p>
              <p className="text-sm text-slate-800">{result.recommendation}</p>
            </CardContent>
          </Card>

          {/* Detailed comparison */}
          <Card>
            <CardHeader
              className="py-3 px-4 cursor-pointer hover:bg-slate-50"
              onClick={() => setShowDetail((v) => !v)}
            >
              <CardTitle className="text-sm flex items-center justify-between">
                Detailed comparison
                <span className="text-xs text-slate-400">{showDetail ? "Hide" : "Show"}</span>
              </CardTitle>
            </CardHeader>
            {showDetail && (
              <CardContent>
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">{result.detailed_comparison}</p>
              </CardContent>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
