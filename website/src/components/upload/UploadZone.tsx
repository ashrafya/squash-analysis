"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Film, X, AlertCircle } from "lucide-react";
import Button from "@/components/ui/Button";
import clsx from "clsx";

const MAX_SIZE_MB = 500;

export default function UploadZone({ onFile }: { onFile: (file: File) => void }) {
  const [error,    setError]    = useState<string | null>(null);
  const [selected, setSelected] = useState<File | null>(null);

  const onDrop = useCallback((accepted: File[], rejected: { errors: { message: string }[] }[]) => {
    setError(null);
    if (rejected.length > 0) { setError(rejected[0]?.errors[0]?.message ?? "Invalid file."); return; }
    if (accepted.length > 0) setSelected(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "video/mp4": [".mp4"] },
    maxFiles: 1, maxSize: MAX_SIZE_MB * 1024 * 1024, multiple: false,
  });

  const fmt = (b: number) => b < 1024*1024 ? `${(b/1024).toFixed(0)} KB` : `${(b/(1024*1024)).toFixed(1)} MB`;

  return (
    <div className="space-y-4">
      {!selected ? (
        <div {...getRootProps()} className={clsx(
          "border-2 p-12 text-center cursor-pointer transition-all duration-150",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lime focus-visible:ring-offset-2 focus-visible:ring-offset-bg-dark",
          isDragActive
            ? "border-lime bg-lime/10 scale-[1.01] shadow-block-lime"
            : "border-white/20 bg-surface-dark hover:border-lime/60 hover:bg-lime/5"
        )}>
          <input {...getInputProps()} aria-label="Upload MP4"/>
          <Upload className={clsx("w-12 h-12 mx-auto mb-4", isDragActive ? "text-lime" : "text-muted")} strokeWidth={1.5}/>
          <p className="font-heading font-bold text-foreground text-xl mb-1">
            {isDragActive ? "Drop it here" : "Drag & drop your MP4"}
          </p>
          <p className="font-body text-muted text-sm mb-6">or click to browse — max {MAX_SIZE_MB} MB</p>
          <Button variant="secondary" size="sm" type="button">Browse files</Button>
        </div>
      ) : (
        <div className="border-2 border-lime bg-surface-dark p-5 flex items-center gap-4 shadow-block-lime">
          <div className="w-12 h-12 bg-lime border-2 border-ink flex items-center justify-center flex-shrink-0">
            <Film className="w-6 h-6 text-ink" strokeWidth={2}/>
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-heading font-semibold text-foreground truncate">{selected.name}</p>
            <p className="font-body text-sm text-muted mt-0.5">{fmt(selected.size)}</p>
          </div>
          <button onClick={() => setSelected(null)} className="text-muted hover:text-foreground transition-colors cursor-pointer p-1" aria-label="Remove">
            <X className="w-5 h-5"/>
          </button>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-coral text-sm font-body" role="alert">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0"/><span>{error}</span>
        </div>
      )}

      {selected && (
        <Button size="lg" variant="lime" className="w-full" onClick={() => onFile(selected)}>
          Start Analysis →
        </Button>
      )}
    </div>
  );
}
