import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FileText, Upload, LayoutDashboard, ShieldCheck,
  LogOut, Trash2, ChevronRight, Clock, Plus
} from "lucide-react";
import { cn, formatRelative, formatBytes } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { getInitials } from "@/lib/utils";

export default function Sidebar({
  documents = [],
  selectedDocId,
  onSelectDoc,
  onUploadClick,
  onDeleteDoc,
  isAdmin,
}) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [hoveredId, setHoveredId] = useState(null);

  const handleSignOut = async () => {
    await signOut();
    navigate("/login");
  };

  return (
    <aside className="flex flex-col w-64 bg-slate-900 text-slate-100 h-screen overflow-hidden shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 border-b border-slate-800">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600">
          <FileText className="h-4 w-4 text-white" />
        </div>
        <div>
          <span className="font-semibold text-white text-sm">DocIntel</span>
          <p className="text-[10px] text-slate-500 leading-tight">Document Intelligence</p>
        </div>
      </div>

      {/* New upload button */}
      <div className="px-3 pt-4 pb-2">
        <button
          onClick={onUploadClick}
          className="flex w-full items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500 transition-colors"
        >
          <Plus className="h-4 w-4" />
          Upload Document
        </button>
      </div>

      {/* Nav links */}
      <nav className="px-3 pb-2">
        {[
          { label: "Dashboard", icon: LayoutDashboard, path: "/dashboard" },
          ...(isAdmin ? [{ label: "Admin", icon: ShieldCheck, path: "/admin" }] : []),
        ].map(({ label, icon: Icon, path }) => (
          <button
            key={path}
            onClick={() => navigate(path)}
            className={cn(
              "flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
              location.pathname === path
                ? "bg-slate-800 text-white"
                : "text-slate-400 hover:bg-slate-800 hover:text-white"
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </button>
        ))}
      </nav>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-3 pb-2">
        <p className="px-3 py-2 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
          Documents
        </p>
        {documents.length === 0 ? (
          <p className="px-3 py-2 text-xs text-slate-500 italic">No documents yet</p>
        ) : (
          <div className="flex flex-col gap-0.5">
            {documents.map((doc) => {
              const name = doc.original_filename || doc.filename || "Untitled";
              const isSelected = doc.id === selectedDocId;
              return (
                <div
                  key={doc.id}
                  className="relative group"
                  onMouseEnter={() => setHoveredId(doc.id)}
                  onMouseLeave={() => setHoveredId(null)}
                >
                  <button
                    onClick={() => onSelectDoc(doc)}
                    className={cn(
                      "flex w-full items-start gap-2.5 rounded-lg px-3 py-2 text-left transition-colors pr-8",
                      isSelected
                        ? "bg-indigo-600 text-white"
                        : "text-slate-300 hover:bg-slate-800 hover:text-white"
                    )}
                  >
                    <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium leading-tight">{name}</p>
                      <p className={cn("text-[10px] mt-0.5", isSelected ? "text-indigo-200" : "text-slate-500")}>
                        {doc.page_count ?? "?"} pages · {formatBytes(doc.file_size)}
                      </p>
                    </div>
                  </button>
                  {hoveredId === doc.id && (
                    <button
                      onClick={(e) => { e.stopPropagation(); onDeleteDoc(doc.id); }}
                      className="absolute right-2 top-2.5 rounded p-0.5 text-slate-500 hover:text-red-400 hover:bg-slate-700 transition-colors"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* User section */}
      <div className="border-t border-slate-800 px-3 py-3">
        <div className="flex items-center gap-2.5">
          <Avatar className="h-7 w-7 shrink-0">
            <AvatarFallback className="bg-indigo-700 text-white text-[10px]">
              {getInitials(user?.email)}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="truncate text-xs font-medium text-slate-200">{user?.email}</p>
            {isAdmin && <p className="text-[10px] text-indigo-400 font-medium">Admin</p>}
          </div>
          <button
            onClick={handleSignOut}
            className="shrink-0 rounded p-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
            title="Sign out"
          >
            <LogOut className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </aside>
  );
}
