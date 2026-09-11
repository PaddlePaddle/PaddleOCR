import React, { useState } from 'react';
import { Copy, Check, Download, Code } from 'lucide-react';
import type { DocumentItem } from '../types';

interface JsonResultViewerProps {
  document: DocumentItem;
  onCopyToast?: (msg: string) => void;
}

export const JsonResultViewer: React.FC<JsonResultViewerProps> = ({ document, onCopyToast }) => {
  const [copied, setCopied] = useState(false);
  const jsonString = JSON.stringify(document, null, 2);

  const handleCopyJson = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    if (onCopyToast) onCopyToast('Raw JSON payload copied to clipboard');
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadJson = () => {
    const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download = `${document.filename.replace(/\.[^/.]+$/, '')}_ocr_result.json`;
    a.click();
    URL.revokeObjectURL(url);
    if (onCopyToast) onCopyToast('JSON file downloaded');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          <Code size={16} />
          <span>Complete Serialized Analysis Document</span>
        </div>

        <div style={{ display: 'flex', gap: '0.6rem' }}>
          <button className="btn btn-secondary" onClick={handleCopyJson} style={{ padding: '0.45rem 0.85rem' }}>
            {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
            <span>{copied ? 'Copied JSON' : 'Copy JSON'}</span>
          </button>

          <button className="btn btn-secondary" onClick={handleDownloadJson} style={{ padding: '0.45rem 0.85rem' }}>
            <Download size={14} />
            <span>Download .json</span>
          </button>
        </div>
      </div>

      <pre className="json-display-pre">{jsonString}</pre>
    </div>
  );
};
