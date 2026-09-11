import React, { useState } from 'react';
import { Copy, Check, ShieldCheck, AlertCircle } from 'lucide-react';
import type { DocumentItem } from '../types';

interface ExtractedFieldsProps {
  document: DocumentItem;
  onCopyToast?: (msg: string) => void;
}

export const ExtractedFields: React.FC<ExtractedFieldsProps> = ({ document, onCopyToast }) => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const fields = document.extracted_fields || document.fields || {};
  const confidences = document.field_confidences || {};
  const fieldKeys = Object.keys(fields).filter((k) => fields[k] !== null && fields[k] !== undefined);

  const handleCopy = (key: string, val: any) => {
    const textVal = Array.isArray(val) ? val.join(', ') : typeof val === 'object' && val !== null ? JSON.stringify(val) : String(val);
    navigator.clipboard.writeText(textVal);
    setCopiedKey(key);
    if (onCopyToast) onCopyToast(`Copied ${key.replace(/_/g, ' ')} to clipboard`);
    setTimeout(() => setCopiedKey(null), 1800);
  };

  const formatKeyName = (key: string) => {
    return key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  };

  if (fieldKeys.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
        <AlertCircle size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.6 }} />
        <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>No structured fields detected</div>
        <div style={{ fontSize: '0.85rem' }}>
          Document type is unclassified or no regex patterns matched. View the "Extracted Text" tab for full raw text.
        </div>
      </div>
    );
  }

  return (
    <div>
      {document.cross_check && (
        <div
          style={{
            marginBottom: '1.25rem',
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: document.cross_check.match ? 'var(--success-bg)' : 'var(--warning-bg)',
            border: `1px solid ${document.cross_check.match ? 'var(--success-border)' : 'var(--warning-border)'}`,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>
            <ShieldCheck size={18} color={document.cross_check.match ? '#34d399' : '#fbbf24'} />
            <span>Cross-Check Verification Score: {Math.round(document.cross_check.score * 100)}%</span>
          </div>
          {document.cross_check.discrepancies?.length > 0 && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#fca5a5' }}>
              Discrepancies: {document.cross_check.discrepancies.join(', ')}
            </div>
          )}
        </div>
      )}

      <div className="fields-grid">
        {fieldKeys.map((key) => {
          const value = fields[key];
          const conf = confidences[key];

          return (
            <div key={key} className="field-card">
              <div>
                <div className="field-top">
                  <span className="field-label">{formatKeyName(key)}</span>
                  {conf !== undefined && (
                    <span
                      className="field-conf-badge"
                      style={{
                        color: conf >= 0.85 ? '#34d399' : conf >= 0.65 ? '#fbbf24' : '#f87171',
                      }}
                    >
                      {Math.round(conf * 100)}%
                    </span>
                  )}
                </div>

                <div className="field-value-row">
                  <div className="field-value" style={{ flex: 1 }}>
                    {Array.isArray(value) ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.2rem' }}>
                        {value.map((item, idx) => (
                          <span
                            key={idx}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              padding: '0.15rem 0.5rem',
                              borderRadius: '4px',
                              backgroundColor: 'rgba(59, 130, 246, 0.15)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              color: 'var(--text-primary)',
                              fontSize: '0.82rem',
                              fontWeight: 500,
                            }}
                          >
                            {String(item)}
                          </span>
                        ))}
                      </div>
                    ) : typeof value === 'object' && value !== null ? (
                      <pre style={{ margin: 0, fontSize: '0.8rem', whiteSpace: 'pre-wrap', maxHeight: '140px', overflowY: 'auto' }}>
                        {JSON.stringify(value, null, 2)}
                      </pre>
                    ) : (
                      String(value)
                    )}
                  </div>
                  <button
                    className="copy-btn"
                    onClick={() => handleCopy(key, value)}
                    title="Copy field value"
                  >
                    {copiedKey === key ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
