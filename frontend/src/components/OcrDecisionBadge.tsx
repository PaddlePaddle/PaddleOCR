import React from 'react';
import { Zap, Scan, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import type { DocumentItem, EngineInfo } from '../types';

interface OcrDecisionBadgeProps {
  document: DocumentItem;
  engineInfo?: EngineInfo | null;
}

export const OcrDecisionBadge: React.FC<OcrDecisionBadgeProps> = ({ document, engineInfo }) => {
  const isBypassed = !document.ocr_required;
  const isError = document.status === 'error' || document.status === 'failed';

  const engineName =
    document.text_source === 'rapid_ocr'
      ? 'RapidOCR'
      : document.text_source === 'paddle_ocr'
      ? 'PaddleOCR'
      : engineInfo?.display_name || 'OCR Engine';

  return (
    <div
      className={`ocr-decision-banner ${
        isError ? 'error' : isBypassed ? 'bypassed' : 'paddle'
      }`}
      style={
        isError
          ? {
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              borderColor: 'rgba(239, 68, 68, 0.3)',
            }
          : undefined
      }
    >
      <div className="decision-left">
        <div
          className="decision-icon"
          style={{
            backgroundColor: isError
              ? 'rgba(239, 68, 68, 0.2)'
              : isBypassed
              ? 'rgba(16, 185, 129, 0.2)'
              : 'rgba(99, 102, 241, 0.2)',
            color: isError ? '#ef4444' : isBypassed ? '#34d399' : '#818cf8',
          }}
        >
          {isError ? <AlertCircle size={22} /> : isBypassed ? <Zap size={22} /> : <Scan size={22} />}
        </div>
        <div>
          <div
            className="decision-title"
            style={{
              color: isError ? '#ef4444' : isBypassed ? '#34d399' : '#a5b4fc',
            }}
          >
            {isError
              ? '❌ OCR Extraction Failed — Document Unreadable'
              : isBypassed
              ? '⚡ Text Layer Detected — OCR Pass Bypassed'
              : `🔍 Scanned Document — ${engineName} Executed`}
          </div>
          <div className="decision-desc">
            {isError
              ? (document.reason || 'The engine returned empty text or zero confidence. Please ensure the document is clear, correctly rotated, and high contrast.')
              : isBypassed
              ? 'PDF contains a clean digital text layer (≥ 50 characters). Extracted directly via Poppler pdftotext with zero character distortion.'
              : `Raster image or scanned document without embedded text (< 50 characters). Deep learning OCR pass executed via ${engineName} with line confidence scoring.`}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <div
          style={{
            padding: '0.35rem 0.75rem',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            fontSize: '0.78rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          Confidence: <strong>{Math.round(document.confidence * 100)}%</strong>
        </div>

        <div
          style={{
            padding: '0.35rem 0.75rem',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            fontSize: '0.78rem',
          }}
        >
          {document.pages} {document.pages === 1 ? 'Page' : 'Pages'}
        </div>

        {(document.checksum_valid !== undefined || document.status === 'warning') && (
          <div
            style={{
              padding: '0.35rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: document.checksum_valid ? 'var(--success-bg)' : 'var(--warning-bg)',
              color: document.checksum_valid ? '#34d399' : '#fbbf24',
              border: `1px solid ${document.checksum_valid ? 'var(--success-border)' : 'var(--warning-border)'}`,
              fontSize: '0.78rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            {document.checksum_valid ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
            {document.checksum_valid
              ? 'Checksums Verified'
              : (document.checksum_reason || document.reason || 'Checksum Warning')}
          </div>
        )}
      </div>
    </div>
  );
};
