import React from 'react';
import { Files, Zap, Scan, CheckCircle2 } from 'lucide-react';
import type { DashboardStats as StatsType, EngineInfo, DocumentItem } from '../types';

interface DashboardStatsProps {
  stats: StatsType;
  onNavigateTab: (tab: 'upload' | 'repository') => void;
  engineInfo?: EngineInfo | null;
  documents?: DocumentItem[];
}

export const DashboardStats: React.FC<DashboardStatsProps> = ({
  stats,
  onNavigateTab,
  engineInfo,
  documents,
}) => {
  const checksumDocs =
    documents?.filter(
      (d) => d.checksum_valid !== undefined && d.checksum_valid !== null
    ) || [];

  const hasChecksumMetrics = checksumDocs.length > 0;
  const checksumValidCount = checksumDocs.filter((d) => d.checksum_valid === true).length;
  const verificationRate = hasChecksumMetrics
    ? Math.round((checksumValidCount / checksumDocs.length) * 100)
    : stats.total > 0
    ? Math.round((stats.completed / stats.total) * 100)
    : 100;

  const ocrLabel = engineInfo
    ? `${engineInfo.display_name || (engineInfo.active_engine === 'rapidocr' ? 'RapidOCR' : engineInfo.active_engine || 'Neural OCR')} Executed`
    : 'Neural OCR Executed';

  return (
    <div className="stats-grid">
      <div
        className="stat-card"
        style={{ cursor: 'pointer' }}
        onClick={() => onNavigateTab('repository')}
      >
        <div className="stat-content">
          <span className="stat-label">Total Vault Documents</span>
          <span className="stat-value">{stats.total}</span>
          <span className="stat-subtext">Processed & indexed</span>
        </div>
        <div className="stat-icon-wrapper" style={{ backgroundColor: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
          <Files size={22} />
        </div>
      </div>

      <div className="stat-card glow-emerald">
        <div className="stat-content">
          <span className="stat-label">Text Layer (OCR Bypassed)</span>
          <span className="stat-value" style={{ color: '#34d399' }}>{stats.ocr_not_required}</span>
          <span className="stat-subtext">Fast digital PDF text extraction</span>
        </div>
        <div className="stat-icon-wrapper" style={{ backgroundColor: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
          <Zap size={22} />
        </div>
      </div>

      <div className="stat-card glow-indigo">
        <div className="stat-content">
          <span className="stat-label">{ocrLabel}</span>
          <span className="stat-value" style={{ color: '#a5b4fc' }}>{stats.ocr_processed}</span>
          <span className="stat-subtext">Scanned documents & image OCR</span>
        </div>
        <div className="stat-icon-wrapper" style={{ backgroundColor: 'rgba(168, 85, 247, 0.15)', color: '#c084fc' }}>
          <Scan size={22} />
        </div>
      </div>

      <div
        className="stat-card"
        title={
          hasChecksumMetrics
            ? 'Percentage of documents with algorithmic checksums (Verhoeff, PAN format, IFSC) that verified valid'
            : 'Percentage of uploaded documents that processed without error'
        }
      >
        <div className="stat-content">
          <span className="stat-label">
            {hasChecksumMetrics ? 'Checksum Verification' : 'Ingestion Success Rate'}
          </span>
          <span className="stat-value">{verificationRate}%</span>
          <span className="stat-subtext">
            {hasChecksumMetrics
              ? `${checksumValidCount} of ${checksumDocs.length} checksums valid`
              : `${stats.completed} of ${stats.total} ingested cleanly`}
          </span>
        </div>
        <div className="stat-icon-wrapper" style={{ backgroundColor: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
          <CheckCircle2 size={22} />
        </div>
      </div>
    </div>
  );
};
