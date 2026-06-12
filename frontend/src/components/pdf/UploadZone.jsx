import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, Loader2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadDocument } from "@/lib/api";

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-excel": [".xls"],
  "text/csv": [".csv"],
  "text/plain": [".txt", ".md"],
  "text/markdown": [".md"],
  "text/html": [".html", ".htm"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/bmp": [".bmp"],
  "image/tiff": [".tiff"],
  "image/gif": [".gif"],
};

const FORMAT_HINT = "PDF, Word, PowerPoint, Excel, CSV, TXT, HTML, Images";

export default function UploadZone({ onSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0];
    if (!file) return;
    setError("");
    setUploading(true);
    setProgress(5);
    try {
      const result = await uploadDocument(file, setProgress);
      setProgress(100);
      if (result.status === "error") {
        setError(result.message || "Processing failed.");
      } else {
        onSuccess?.(result);
      }
    } catch (err) {
      setError(err.userMessage || "Upload failed. Please try again.");
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 600);
    }
  }, [onSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: 50 * 1024 * 1024,
    multiple: false,
    disabled: uploading,
  });

  return (
    <div className="p-8 flex flex-col items-center justify-center h-full">
      <div
        {...getRootProps()}
        className={cn(
          "flex flex-col items-center justify-center w-full max-w-lg rounded-2xl border-2 border-dashed transition-all cursor-pointer p-10 text-center",
          isDragActive
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50 bg-white",
          uploading && "pointer-events-none opacity-70"
        )}
      >
        <input {...getInputProps()} />

        {uploading ? (
          <>
            <Loader2 className="h-10 w-10 text-indigo-600 animate-spin mb-4" />
            <p className="text-sm font-medium text-slate-700">Processing document...</p>
            <p className="text-xs text-slate-400 mt-1">Extracting, chunking and embedding</p>
            {progress > 0 && (
              <div className="mt-4 w-full bg-slate-200 rounded-full h-1.5">
                <div
                  className="bg-indigo-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-indigo-50 mb-4">
              {isDragActive
                ? <FileText className="h-7 w-7 text-indigo-600" />
                : <Upload className="h-7 w-7 text-indigo-600" />}
            </div>
            <p className="text-base font-semibold text-slate-800 mb-1">
              {isDragActive ? "Drop your document here" : "Upload a Document"}
            </p>
            <p className="text-sm text-slate-500">Drag & drop or click to browse</p>
            <p className="text-xs text-slate-400 mt-2">{FORMAT_HINT} · max 50 MB</p>
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700 max-w-lg w-full">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
