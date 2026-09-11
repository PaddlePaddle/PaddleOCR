import React from 'react';
import { Check, Loader2 } from 'lucide-react';
import type { UploadProgress } from '../types';

interface ProcessingTimelineProps {
  progress: UploadProgress;
}

const STEPS = [
  { id: 'uploading', label: '1. Ingesting Document', sub: 'Streaming payload to backend' },
  { id: 'analyzing', label: '2. Detecting Text Layer', sub: 'Checking Poppler pdftotext (>=50 chars)' },
  { id: 'extracting', label: '3. Extraction / OCR', sub: 'Running Neural OCR or Native Text Extractor' },
  { id: 'verifying', label: '4. Checksums & PII Guard', sub: 'Verhoeff, PAN format, IFSC, masking' },
  { id: 'done', label: '5. Stored & Indexed', sub: 'Ready in document repository' },
];

export const ProcessingTimeline: React.FC<ProcessingTimelineProps> = ({ progress }) => {
  const getStepStatus = (stepId: string) => {
    const stepOrder = ['uploading', 'analyzing', 'extracting', 'verifying', 'done'];
    const currentIndex = stepOrder.indexOf(progress.step);
    const stepIndex = stepOrder.indexOf(stepId);

    if (progress.step === 'done') return 'done';
    if (stepIndex < currentIndex) return 'done';
    if (stepIndex === currentIndex) return 'current';
    return 'pending';
  };

  return (
    <div style={{ padding: '1.5rem', backgroundColor: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', margin: '1.5rem 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center' }}>
        <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Live Pipeline Progress</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: '#818cf8' }}>
          {progress.percent}%
        </span>
      </div>

      <div style={{ height: '6px', background: 'var(--bg-base)', borderRadius: '4px', overflow: 'hidden', marginBottom: '1.5rem' }}>
        <div
          style={{
            height: '100%',
            width: `${progress.percent}%`,
            background: 'linear-gradient(90deg, #6366f1, #06b6d4)',
            transition: 'width 0.3s ease',
          }}
        />
      </div>

      <div className="timeline-container">
        {STEPS.map((s) => {
          const status = getStepStatus(s.id);
          return (
            <div key={s.id} className="timeline-step">
              <div className={`timeline-bullet ${status}`}>
                {status === 'done' ? (
                  <Check size={14} />
                ) : status === 'current' ? (
                  <Loader2 size={14} className="spin-anim" />
                ) : (
                  <span>•</span>
                )}
              </div>
              <div>
                <div className="timeline-label">{s.label}</div>
                <div className="timeline-sub">{s.sub}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
