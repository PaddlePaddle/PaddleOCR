import React, { useState, useRef } from 'react';
import { UploadCloud, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import type { SupportedType, DocumentItem, UploadProgress } from '../types';
import { api } from '../services/api';
import { ProcessingTimeline } from './ProcessingTimeline';

interface UploadCardProps {
  supportedTypes: SupportedType[];
  onUploadSuccess: (doc: DocumentItem) => void;
  onError: (msg: string) => void;
}

export const UploadCard: React.FC<UploadCardProps> = ({
  supportedTypes,
  onUploadSuccess,
  onError,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedType, setSelectedType] = useState<string>('auto');
  const [expectedData, setExpectedData] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<UploadProgress>({
    step: 'idle',
    percent: 0,
    message: '',
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    const validExtensions = ['.pdf', '.png', '.jpg', '.jpeg', '.webp'];
    const hasValidExt = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExt) {
      onError(`Invalid file type. Please upload a PDF, PNG, JPG, or WEBP.`);
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      onError(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Maximum size is 25MB.`);
      return;
    }

    setIsProcessing(true);
    setProgress({ step: 'uploading', percent: 20, message: 'Uploading document payload...' });

    try {
      // Step simulation for rich UX
      setTimeout(() => {
        setProgress({ step: 'analyzing', percent: 45, message: 'Checking embedded text layer (pdftotext)...' });
      }, 300);

      setTimeout(() => {
        setProgress({ step: 'extracting', percent: 70, message: 'Classifying document & running extraction...' });
      }, 700);

      setTimeout(() => {
        setProgress({ step: 'verifying', percent: 90, message: 'Validating checksums & masking PII...' });
      }, 1000);

      const doc = await api.uploadDocument(file, selectedType, expectedData || undefined);

      setProgress({ step: 'done', percent: 100, message: 'Document analysis complete!' });
      setTimeout(() => {
        setIsProcessing(false);
        onUploadSuccess(doc);
      }, 400);
    } catch (err: any) {
      setIsProcessing(false);
      setProgress({ step: 'error', percent: 0, message: err.message || 'Processing failed' });
      onError(err.message || 'Upload and processing failed');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="upload-card">
      <div className="upload-header">
        <div>
          <h2 className="upload-title">Ingest & Verify Document</h2>
          <p className="upload-subtitle">
            Upload any Indian ID, financial document, or certificate for automated classification, text layer detection, and checksum verification.
          </p>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg,.webp"
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
        disabled={isProcessing}
      />

      {isProcessing ? (
        <ProcessingTimeline progress={progress} />
      ) : (
        <>
          <div
            className={`dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">
              <UploadCloud size={28} />
            </div>
            <div className="dropzone-text">Click to browse or drag & drop files here</div>
            <div className="dropzone-subtext">PDF documents (digital or scanned), PNG, JPG up to 25MB</div>
            <div className="file-types-badge-row">
              <span className="file-badge">PDF (Auto text layer detection)</span>
              <span className="file-badge">PNG</span>
              <span className="file-badge">JPG / JPEG</span>
              <span className="file-badge">WEBP</span>
            </div>
          </div>

          <div className="upload-controls-grid">
            <div className="form-group">
              <label className="form-label">Document Classification</label>
              <select
                className="form-select"
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                disabled={isProcessing}
              >
                <option value="auto">⚡ Auto-Detect Type (AI Classifier)</option>
                {supportedTypes
                  .filter((t) => t.id !== 'auto')
                  .map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name} ({t.category})
                    </option>
                  ))}
              </select>
            </div>

            <div className="form-group" style={{ justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ alignSelf: 'flex-start', marginTop: 'auto' }}
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <Sparkles size={15} />
                <span>{showAdvanced ? 'Hide Advanced Options' : 'Cross-Check Verification (Optional)'}</span>
                {showAdvanced ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>
            </div>
          </div>

          {showAdvanced && (
            <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
              <div className="form-group">
                <label className="form-label">Expected Data JSON (For Automated Cross-Check)</label>
                <textarea
                  className="form-textarea"
                  rows={3}
                  placeholder='{"name": "VIKRAM SHARMA", "pan": "ABCDE1234F", "dob": "15/08/1985"}'
                  value={expectedData}
                  onChange={(e) => setExpectedData(e.target.value)}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
                  Provide expected customer records to test fuzzy cross-checking and discrepancy calculation.
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
