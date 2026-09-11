import React, { useState, useMemo } from 'react';
import { Search, Copy, Check, Download } from 'lucide-react';

interface ExtractedTextViewerProps {
  text: string;
  filename: string;
  onCopyToast?: (msg: string) => void;
}

export const ExtractedTextViewer: React.FC<ExtractedTextViewerProps> = ({
  text,
  filename,
  onCopyToast,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [copied, setCopied] = useState(false);

  const handleCopyAll = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    if (onCopyToast) onCopyToast('All extracted text copied to clipboard');
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadTxt = () => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename.replace(/\.[^/.]+$/, '')}_extracted.txt`;
    a.click();
    URL.revokeObjectURL(url);
    if (onCopyToast) onCopyToast('Text file downloaded');
  };

  // Highlights search matches in text
  const renderedText = useMemo(() => {
    if (!searchQuery.trim()) {
      return text;
    }

    const query = searchQuery.trim();
    // Escape regex characters
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escaped})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, idx) =>
      regex.test(part) ? (
        <mark key={idx} className="highlight-match">
          {part}
        </mark>
      ) : (
        part
      )
    );
  }, [text, searchQuery]);

  const matchCount = useMemo(() => {
    if (!searchQuery.trim()) return 0;
    const query = searchQuery.toLowerCase().trim();
    const matches = (text.toLowerCase().match(new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
    return matches;
  }, [text, searchQuery]);

  return (
    <div className="text-viewer-box">
      <div className="text-viewer-toolbar">
        <div className="search-input-wrap">
          <Search size={15} className="search-icon-pos" />
          <input
            type="text"
            className="form-input"
            placeholder="Search within extracted text..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {searchQuery && (
            <span style={{ fontSize: '0.8rem', color: '#fbbf24', fontWeight: 500 }}>
              {matchCount} {matchCount === 1 ? 'match' : 'matches'}
            </span>
          )}

          <button className="btn btn-secondary" onClick={handleCopyAll} style={{ padding: '0.45rem 0.85rem' }}>
            {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy All'}</span>
          </button>

          <button className="btn btn-secondary" onClick={handleDownloadTxt} style={{ padding: '0.45rem 0.85rem' }}>
            <Download size={14} />
            <span>Download .txt</span>
          </button>
        </div>
      </div>

      <pre className="text-display-pre">{renderedText || 'No text extracted from document.'}</pre>

      <div style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', textAlign: 'right' }}>
        Total Characters: {text.length} • Lines: {text.split('\n').length}
      </div>
    </div>
  );
};
