import React, { useState } from 'react';
import { Search, Eye, Trash2, Download, FileText, Zap, Scan, RefreshCw, AlertCircle, AlertTriangle } from 'lucide-react';
import type { DocumentItem, SupportedType, EngineInfo } from '../types';
import { api } from '../services/api';

interface DocumentsTableProps {
  documents: DocumentItem[];
  supportedTypes: SupportedType[];
  onSelectDocument: (doc: DocumentItem) => void;
  onDeleteDocument: (id: string) => void;
  onRefresh: () => void;
  isLoading?: boolean;
  engineInfo?: EngineInfo | null;
}

export const DocumentsTable: React.FC<DocumentsTableProps> = ({
  documents,
  supportedTypes,
  onSelectDocument,
  onDeleteDocument,
  onRefresh,
  isLoading = false,
  engineInfo,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');

  const filteredDocs = documents.filter((doc) => {
    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase();
      const matchName = doc.filename.toLowerCase().includes(q);
      const matchType = (doc.document_type || doc.doc_type).toLowerCase().includes(q);
      const matchText = (doc.extracted_text || '').toLowerCase().includes(q);
      if (!matchName && !matchType && !matchText) return false;
    }

    if (typeFilter !== 'all' && doc.doc_type.toLowerCase() !== typeFilter.toLowerCase()) {
      return false;
    }

    if (sourceFilter === 'bypassed' && doc.ocr_required !== false) {
      return false;
    }
    if (sourceFilter === 'ocr' && doc.ocr_required !== true) {
      return false;
    }

    return true;
  });

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="table-card">
      <div className="table-toolbar">
        <div className="table-filter-group">
          <div className="search-input-wrap">
            <Search size={15} className="search-icon-pos" />
            <input
              type="text"
              className="form-input"
              placeholder="Search filename, type, or extracted text..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: '280px' }}
            />
          </div>

          <select
            className="form-select"
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            style={{ width: '180px' }}
          >
            <option value="all">All Document Types</option>
            {supportedTypes
              .filter((t) => t.id !== 'auto')
              .map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
          </select>

          <select
            className="form-select"
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            style={{ width: '190px' }}
          >
            <option value="all">All Ingestion Modes</option>
            <option value="bypassed">⚡ Text Layer (OCR Bypassed)</option>
            <option value="ocr">🔍 OCR Executed</option>
          </select>
        </div>

        <button
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={isLoading}
          style={{ padding: '0.5rem 0.85rem' }}
        >
          <RefreshCw size={14} className={isLoading ? 'spin-anim' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="table-wrapper">
        <table className="doc-table">
          <thead>
            <tr>
              <th style={{ width: '60px' }}>Preview</th>
              <th>Document Name</th>
              <th>Document Type</th>
              <th>OCR Decision</th>
              <th>Confidence</th>
              <th>Processed At</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredDocs.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '3.5rem 1rem', color: 'var(--text-muted)' }}>
                  <FileText size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.5 }} />
                  <div style={{ fontWeight: 600 }}>No documents found</div>
                  <div style={{ fontSize: '0.85rem' }}>Upload a new document or adjust your filters.</div>
                </td>
              </tr>
            ) : (
              filteredDocs.map((doc) => {
                const isBypassed = !doc.ocr_required;
                const isError = doc.status === 'error' || doc.status === 'failed';
                const previewUrl = api.getPreviewUrl(doc.id);

                const getEngineLabel = () => {
                  if (isBypassed) return 'Text Layer';
                  if (doc.text_source === 'rapid_ocr') return 'RapidOCR';
                  if (doc.text_source === 'paddle_ocr') return 'PaddleOCR';
                  if (engineInfo?.display_name) return engineInfo.display_name;
                  return 'Neural OCR';
                };

                return (
                  <tr
                    key={doc.id}
                    className={isError ? 'row-error' : undefined}
                    style={isError ? { backgroundColor: 'rgba(239, 68, 68, 0.04)' } : undefined}
                  >
                    <td>
                      {doc.has_preview ? (
                        <img
                          src={previewUrl}
                          alt={doc.filename}
                          className="table-thumb"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />
                      ) : (
                        <div className="thumb-fallback">
                          <FileText size={18} />
                        </div>
                      )}
                    </td>

                    <td>
                      <div
                        style={{
                          fontWeight: 600,
                          color: 'var(--text-main)',
                          cursor: 'pointer',
                          maxWidth: '240px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                        onClick={() => onSelectDocument(doc)}
                        title="Click to view full inspection"
                      >
                        {doc.filename}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
                        {(doc.file_size / 1024).toFixed(1)} KB • {doc.pages} {doc.pages === 1 ? 'page' : 'pages'}
                      </div>
                    </td>

                    <td>
                      {isError ? (
                        <span
                          title={doc.reason || 'Image returned no text or zero confidence — unreadable or blank image'}
                          style={{
                            padding: '0.2rem 0.6rem',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '0.78rem',
                            fontWeight: 600,
                            backgroundColor: 'rgba(239, 68, 68, 0.15)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.3rem',
                          }}
                        >
                          <AlertCircle size={13} />
                          OCR Failed
                        </span>
                      ) : doc.status === 'warning' || doc.checksum_valid === false ? (
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                          <span
                            style={{
                              padding: '0.2rem 0.6rem',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: '0.78rem',
                              fontWeight: 500,
                              backgroundColor: 'rgba(255, 255, 255, 0.07)',
                              color: 'var(--text-main)',
                              display: 'inline-block',
                            }}
                          >
                            {doc.document_type || doc.doc_type}
                          </span>
                          <span
                            title={doc.reason || doc.checksum_reason ? `Validation warning: ${doc.reason || doc.checksum_reason}` : 'Checksum/format validation warning'}
                            style={{
                              padding: '0.15rem 0.5rem',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: '0.72rem',
                              fontWeight: 600,
                              backgroundColor: 'var(--warning-bg)',
                              color: '#fbbf24',
                              border: '1px solid var(--warning-border)',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                            }}
                          >
                            <AlertTriangle size={11} />
                            {doc.reason || doc.checksum_reason || 'Warning'}
                          </span>
                        </div>
                      ) : (
                        <span
                          style={{
                            padding: '0.2rem 0.6rem',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '0.78rem',
                            fontWeight: 500,
                            backgroundColor: 'rgba(255, 255, 255, 0.07)',
                            color: 'var(--text-main)',
                            display: 'inline-block',
                          }}
                        >
                          {doc.document_type || doc.doc_type}
                        </span>
                      )}
                    </td>

                    <td>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          padding: '0.2rem 0.6rem',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: '0.78rem',
                          fontWeight: 600,
                          backgroundColor: isError
                            ? 'rgba(239, 68, 68, 0.1)'
                            : isBypassed
                            ? 'var(--success-bg)'
                            : 'rgba(99, 102, 241, 0.15)',
                          color: isError ? '#ef4444' : isBypassed ? '#34d399' : '#818cf8',
                          border: `1px solid ${
                            isError
                              ? 'rgba(239, 68, 68, 0.25)'
                              : isBypassed
                              ? 'var(--success-border)'
                              : 'rgba(99, 102, 241, 0.3)'
                          }`,
                        }}
                      >
                        {isError ? <AlertCircle size={13} /> : isBypassed ? <Zap size={13} /> : <Scan size={13} />}
                        {getEngineLabel()}
                      </span>
                    </td>

                    <td>
                      {isError ? (
                        <span
                          title={doc.reason || 'Zero confidence OCR failure'}
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontWeight: 600,
                            color: '#ef4444',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.25rem',
                          }}
                        >
                          <AlertCircle size={13} />
                          0%
                        </span>
                      ) : (
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-main)' }}>
                          {Math.round(doc.confidence * 100)}%
                        </span>
                      )}
                    </td>

                    <td style={{ fontSize: '0.8rem', color: 'var(--text-subtle)' }}>
                      {formatDate(doc.created_at)}
                    </td>

                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '0.35rem' }}>
                        <button
                          className="btn btn-secondary"
                          onClick={() => onSelectDocument(doc)}
                          title="Inspect Document"
                          style={{ padding: '0.35rem 0.65rem' }}
                        >
                          <Eye size={14} />
                          <span style={{ fontSize: '0.78rem' }}>Inspect</span>
                        </button>

                        <a
                          href={api.getFileUrl(doc.id, true)}
                          download={doc.filename}
                          className="copy-btn"
                          title="Download original file"
                        >
                          <Download size={14} />
                        </a>

                        <button
                          className="copy-btn"
                          onClick={() => {
                            if (window.confirm(`Delete "${doc.filename}" from vault?`)) {
                              onDeleteDocument(doc.id);
                            }
                          }}
                          title="Delete document"
                          style={{ color: '#f87171' }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
