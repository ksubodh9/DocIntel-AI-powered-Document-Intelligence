import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  Users, FileText, Activity, HardDrive,
  ArrowLeft, ChevronRight, ChevronDown, Loader2,
  MessageSquare, Star
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { getAdminStats, getAdminUsers, getAdminUserDocuments, getAdminFeedback } from "@/lib/api";
import { formatDate, formatRelative, formatBytes, getInitials } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

function StatCard({ icon: Icon, label, value, sub, color = "indigo" }) {
  const colors = {
    indigo: "bg-indigo-50 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-400",
    green: "bg-green-50 text-green-600 dark:bg-green-950 dark:text-green-400",
    amber: "bg-amber-50 text-amber-600 dark:bg-amber-950 dark:text-amber-400",
    violet: "bg-violet-50 text-violet-600 dark:bg-violet-950 dark:text-violet-400",
  };
  return (
    <Card>
      <CardContent className="pt-5 flex items-center gap-4">
        <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${colors[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{value ?? <Loader2 className="h-5 w-5 animate-spin text-slate-300" />}</p>
          <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
          {sub && <p className="text-xs text-slate-400 dark:text-slate-500">{sub}</p>}
        </div>
      </CardContent>
    </Card>
  );
}

function Stars({ value = 0 }) {
  if (!value) return <span className="text-xs text-slate-400 dark:text-slate-500">No rating</span>;
  return (
    <span className="inline-flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <Star
          key={n}
          className={n <= value
            ? "h-3.5 w-3.5 fill-amber-400 text-amber-400"
            : "h-3.5 w-3.5 text-slate-300 dark:text-slate-600"}
        />
      ))}
    </span>
  );
}

function FeedbackSection({ feedback, isLoading, stats }) {
  return (
    <Card className="mt-6">
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-sm">Feedback ({stats?.total_feedback ?? feedback.length})</CardTitle>
        <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          {stats?.avg_rating != null && (
            <span className="inline-flex items-center gap-1">
              <Star className="h-3.5 w-3.5 fill-amber-400 text-amber-400" />
              {stats.avg_rating} avg
            </span>
          )}
          {stats?.new_feedback > 0 && (
            <Badge variant="indigo" className="text-[10px]">{stats.new_feedback} new</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-10 text-slate-400 dark:text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" /><span className="text-sm">Loading feedback...</span>
          </div>
        ) : feedback.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-slate-400 dark:text-slate-500">No feedback yet.</p>
        ) : (
          <div className="divide-y dark:divide-slate-800">
            {feedback.map((f) => (
              <div key={f.id} className="px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <Stars value={f.rating} />
                    <span className="truncate text-xs text-slate-500 dark:text-slate-400">
                      {f.email || "anonymous"}
                    </span>
                  </div>
                  <span className="shrink-0 text-xs text-slate-400 dark:text-slate-500">
                    {f.created_at ? formatRelative(f.created_at) : ""}
                  </span>
                </div>
                {f.comment && (
                  <p className="mt-1.5 text-sm text-slate-700 dark:text-slate-300">{f.comment}</p>
                )}
                {f.last_feature_used && (
                  <Badge variant="secondary" className="mt-2 text-[9px]">{f.last_feature_used}</Badge>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function UserRow({ user, onExpand, expanded }) {
  return (
    <>
      <tr
        className="hover:bg-slate-50 cursor-pointer transition-colors dark:hover:bg-slate-800/50"
        onClick={() => onExpand(user.user_id)}
      >
        <td className="px-4 py-3">
          <div className="flex items-center gap-2.5">
            <Avatar className="h-7 w-7 shrink-0">
              <AvatarFallback className="bg-indigo-100 text-indigo-700 text-[10px] dark:bg-indigo-900 dark:text-indigo-300">
                {getInitials(user.email)}
              </AvatarFallback>
            </Avatar>
            <span className="text-sm font-medium text-slate-800 dark:text-slate-200">{user.email}</span>
          </div>
        </td>
        <td className="px-4 py-3 text-sm text-slate-600 dark:text-slate-400">{formatDate(user.joined_at)}</td>
        <td className="px-4 py-3">
          <Badge variant="indigo" className="text-[10px]">{user.documents_count} docs</Badge>
        </td>
        <td className="px-4 py-3">
          <span className="text-sm text-slate-600 dark:text-slate-400">{user.api_calls_count} calls</span>
        </td>
        <td className="px-4 py-3 text-sm text-slate-400 dark:text-slate-500">{user.last_active ? formatRelative(user.last_active) : "—"}</td>
        <td className="px-4 py-3 text-slate-400 dark:text-slate-500">
          {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-slate-50 dark:bg-slate-800/40">
          <td colSpan={6} className="px-4 pb-3">
            <UserDocuments userId={user.user_id} />
          </td>
        </tr>
      )}
    </>
  );
}

function UserDocuments({ userId }) {
  const { data, isLoading } = useQuery({
    queryKey: ["admin-user-docs", userId],
    queryFn: () => getAdminUserDocuments(userId),
  });

  if (isLoading) return <div className="py-3 flex items-center gap-2 text-xs text-slate-400 dark:text-slate-500"><Loader2 className="h-3 w-3 animate-spin" />Loading documents...</div>;
  if (!data?.length) return <p className="py-3 text-xs text-slate-400 dark:text-slate-500 italic">No documents yet.</p>;

  return (
    <div className="flex flex-wrap gap-2 py-2">
      {data.map((doc) => (
        <div key={doc.id} className="flex items-center gap-1.5 rounded-lg border bg-white px-3 py-1.5 text-xs text-slate-600 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300">
          <FileText className="h-3 w-3 text-slate-400" />
          <span className="max-w-[200px] truncate">{doc.original_filename || doc.filename}</span>
          {doc.document_type && <Badge variant="secondary" className="text-[9px] ml-1">{doc.document_type}</Badge>}
          <span className="text-slate-400 dark:text-slate-500">{formatRelative(doc.created_at)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Admin() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [expandedUser, setExpandedUser] = useState(null);

  const { data: stats } = useQuery({ queryKey: ["admin-stats"], queryFn: getAdminStats });
  const { data: users = [], isLoading: usersLoading } = useQuery({ queryKey: ["admin-users"], queryFn: getAdminUsers });
  const { data: feedback = [], isLoading: feedbackLoading } = useQuery({ queryKey: ["admin-feedback"], queryFn: getAdminFeedback });

  const toggleExpand = (id) => setExpandedUser((e) => (e === id ? null : id));

  // Build chart data from users
  const chartData = [...users]
    .sort((a, b) => (b.api_calls_count || 0) - (a.api_calls_count || 0))
    .slice(0, 10)
    .map((u) => ({
      name: u.email.split("@")[0],
      calls: u.api_calls_count || 0,
      docs: u.documents_count || 0,
    }));

  return (
    <div className="flex h-screen flex-col bg-slate-50 overflow-hidden dark:bg-slate-950">
      {/* Header */}
      <div className="flex h-14 items-center gap-3 border-b bg-white px-6 shrink-0 dark:bg-slate-900">
        <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600">
          <FileText className="h-3.5 w-3.5 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">Admin Dashboard</p>
          <p className="text-xs text-slate-400 dark:text-slate-500">{user?.email}</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {/* Stats cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatCard icon={Users} label="Total users" value={stats?.total_users} color="indigo" />
          <StatCard icon={FileText} label="Documents" value={stats?.total_documents} sub={`${stats?.docs_indexed ?? 0} indexed`} color="green" />
          <StatCard icon={Activity} label="API calls" value={stats?.total_api_calls} color="amber" />
          <StatCard icon={HardDrive} label="Storage" value={stats ? formatBytes(stats.storage_used_bytes) : undefined} color="violet" />
        </div>

        {/* Usage chart */}
        {chartData.length > 0 && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="text-sm">Top users by API calls</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Bar dataKey="calls" fill="#6366f1" radius={[4, 4, 0, 0]} name="API calls" />
                  <Bar dataKey="docs" fill="#a5b4fc" radius={[4, 4, 0, 0]} name="Documents" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Users table */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">All users ({users.length})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {usersLoading ? (
              <div className="flex items-center justify-center py-10 gap-2 text-slate-400 dark:text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Loading users...</span>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b text-left">
                      {["User", "Joined", "Documents", "API Calls", "Last active", ""].map((h) => (
                        <th key={h} className="px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {users.map((u) => (
                      <UserRow
                        key={u.user_id}
                        user={u}
                        onExpand={toggleExpand}
                        expanded={expandedUser === u.user_id}
                      />
                    ))}
                    {users.length === 0 && (
                      <tr><td colSpan={6} className="px-4 py-8 text-center text-sm text-slate-400 dark:text-slate-500">No users yet.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feedback */}
        <FeedbackSection feedback={feedback} isLoading={feedbackLoading} stats={stats} />
      </div>
    </div>
  );
}
