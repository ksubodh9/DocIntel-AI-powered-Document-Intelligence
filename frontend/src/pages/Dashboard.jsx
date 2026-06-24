import { useState, useEffect, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LayoutGrid, MessageSquare, Table2, BookOpen, GitCompare } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useAuth } from "@/contexts/AuthContext";
import Sidebar from "@/components/layout/Sidebar";
import Header from "@/components/layout/Header";
import UploadZone from "@/components/pdf/UploadZone";
import InsightsPanel from "@/components/pdf/InsightsPanel";
import ChatPanel from "@/components/pdf/ChatPanel";
import TablesPanel from "@/components/pdf/TablesPanel";
import MultiChatPanel from "@/components/pdf/MultiChatPanel";
import ComparePanel from "@/components/pdf/ComparePanel";
import { listDocuments, deleteDocument } from "@/lib/api";
import ApiKeyDialog from "@/components/byok/ApiKeyDialog";
import { hasChosen } from "@/lib/byok";
import { toast } from "@/components/ui/toaster";

export default function Dashboard() {
  const { isAdmin } = useAuth();
  const queryClient = useQueryClient();
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [activeTab, setActiveTab] = useState("insights");
  const [showUpload, setShowUpload] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [keysOpen, setKeysOpen] = useState(false);

  const { data: documents = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    refetchInterval: selectedDoc ? false : 10_000,
  });

  // Post-login: prompt for AI provider/keys once per session if not chosen yet.
  useEffect(() => {
    if (!hasChosen()) setKeysOpen(true);
  }, []);

  // Auto-select first doc
  useEffect(() => {
    if (!selectedDoc && documents.length > 0) setSelectedDoc(documents[0]);
  }, [documents, selectedDoc]);

  // Sync selected doc with latest data
  useEffect(() => {
    if (selectedDoc) {
      const fresh = documents.find((d) => d.id === selectedDoc.id);
      if (fresh) setSelectedDoc(fresh);
    }
  }, [documents]);

  const handleUploadSuccess = useCallback((result) => {
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    setShowUpload(false);
    // Find the newly uploaded doc and select it
    setSelectedDoc({ id: result.document_id, filename: result.filename, original_filename: result.filename, page_count: result.page_count, file_size: result.file_size, status: result.status });
    toast({ title: "Upload complete", description: `${result.filename} is ready.`, variant: "success" });
  }, [queryClient]);

  const handleDelete = useCallback(async (docId) => {
    try {
      await deleteDocument(docId);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      if (selectedDoc?.id === docId) setSelectedDoc(null);
      toast({ title: "Document deleted", variant: "success" });
    } catch (err) {
      toast({ title: "Delete failed", description: err.userMessage, variant: "error" });
    }
  }, [selectedDoc, queryClient]);

  const handleAskQuestion = useCallback((q) => {
    setPendingQuestion(q);
    setActiveTab("chat");
  }, []);

  // After analysis completes, refresh the document list so InsightsPanel
  // picks up the saved results (document_type, summaries, etc.) if it remounts.
  const handleAnalysisComplete = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["documents"] });
  }, [queryClient]);

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-950">
      <Sidebar
        documents={documents}
        selectedDocId={selectedDoc?.id}
        onSelectDoc={(doc) => { setSelectedDoc(doc); setShowUpload(false); setSidebarOpen(false); }}
        onUploadClick={() => { setShowUpload(true); setSelectedDoc(null); setSidebarOpen(false); }}
        onDeleteDoc={handleDelete}
        isAdmin={isAdmin}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* flex-1 flex-col overflow-hidden min-h-0 lets the Tabs root fill the
          viewport; tabs themselves use absolute inset-0 for reliable sizing */}
      <div className="flex flex-1 flex-col overflow-hidden min-h-0">
        <Header
          document={!showUpload ? selectedDoc : null}
          onMenuClick={() => setSidebarOpen(true)}
          onOpenKeys={() => setKeysOpen(true)}
        />

        {showUpload || !selectedDoc ? (
          <UploadZone onSuccess={handleUploadSuccess} />
        ) : (
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex flex-1 flex-col overflow-hidden min-h-0"
          >
            <div className="border-b bg-white px-3 py-2 flex-shrink-0 overflow-x-auto dark:bg-slate-900 sm:px-5">
              <TabsList className="h-8 w-max">
                <TabsTrigger value="insights" className="text-xs gap-1.5">
                  <LayoutGrid className="h-3.5 w-3.5" /> Insights
                </TabsTrigger>
                <TabsTrigger value="chat" className="text-xs gap-1.5">
                  <MessageSquare className="h-3.5 w-3.5" /> Chat
                </TabsTrigger>
                <TabsTrigger value="tables" className="text-xs gap-1.5">
                  <Table2 className="h-3.5 w-3.5" /> Tables
                </TabsTrigger>
                <TabsTrigger value="multi" className="text-xs gap-1.5">
                  <BookOpen className="h-3.5 w-3.5" /> Multi-PDF
                </TabsTrigger>
                <TabsTrigger value="compare" className="text-xs gap-1.5">
                  <GitCompare className="h-3.5 w-3.5" /> Compare
                </TabsTrigger>
              </TabsList>
            </div>

            {/* forceMount keeps all panels alive so their state (chat history,
                analysis results) survives tab switches.
                data-[state=inactive]:hidden hides the inactive ones visually. */}
            <div className="flex-1 min-h-0 relative">
              <TabsContent
                value="insights"
                forceMount
                className="absolute inset-0 overflow-y-auto m-0 data-[state=inactive]:hidden"
              >
                <InsightsPanel
                  key={selectedDoc.id}
                  document={selectedDoc}
                  onAskQuestion={handleAskQuestion}
                  onAnalysisComplete={handleAnalysisComplete}
                />
              </TabsContent>
              <TabsContent
                value="chat"
                forceMount
                className="absolute inset-0 m-0 flex flex-col min-h-0 data-[state=inactive]:hidden"
              >
                <ChatPanel
                  key={selectedDoc.id}
                  document={selectedDoc}
                  pendingQuestion={pendingQuestion}
                  onPendingClear={() => setPendingQuestion(null)}
                />
              </TabsContent>
              <TabsContent
                value="tables"
                forceMount
                className="absolute inset-0 m-0 overflow-hidden data-[state=inactive]:hidden"
              >
                <TablesPanel key={selectedDoc.id} document={selectedDoc} />
              </TabsContent>
              <TabsContent
                value="multi"
                forceMount
                className="absolute inset-0 m-0 flex flex-col min-h-0 data-[state=inactive]:hidden"
              >
                <MultiChatPanel documents={documents} />
              </TabsContent>
              <TabsContent
                value="compare"
                forceMount
                className="absolute inset-0 overflow-y-auto m-0 data-[state=inactive]:hidden"
              >
                <ComparePanel documents={documents} />
              </TabsContent>
            </div>
          </Tabs>
        )}
      </div>

      <ApiKeyDialog open={keysOpen} onClose={() => setKeysOpen(false)} />
    </div>
  );
}
