import React from 'react';
import { FileText, UploadCloud, FolderArchive, BarChart3, Settings } from 'lucide-react';

import type { EngineInfo } from '../types';

export type NavTab = 'dashboard' | 'upload' | 'repository' | 'analytics';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  onOpenSettings: () => void;
  engineInfo?: EngineInfo | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onOpenSettings,
  engineInfo,
}) => {
  const engineDisplayName =
    engineInfo?.display_name ||
    (engineInfo?.active_engine === 'rapidocr'
      ? 'RapidOCR'
      : engineInfo?.active_engine === 'paddleocr'
      ? 'PaddleOCR'
      : engineInfo?.engine || 'RapidOCR');
  const deviceName = engineInfo?.device?.toUpperCase() || 'CPU';

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <FileText size={20} />
        </div>
        <div>
          <div className="brand-title">DocuScan AI</div>
          <div className="brand-subtitle">
            {engineInfo ? `${engineDisplayName} Engine` : 'OCR Engine'}
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <button
          className={`nav-item ${currentTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => onSelectTab('dashboard')}
        >
          <BarChart3 size={18} />
          <span>Dashboard</span>
        </button>

        <button
          className={`nav-item ${currentTab === 'upload' ? 'active' : ''}`}
          onClick={() => onSelectTab('upload')}
        >
          <UploadCloud size={18} />
          <span>Upload & Verify</span>
        </button>

        <button
          className={`nav-item ${currentTab === 'repository' ? 'active' : ''}`}
          onClick={() => onSelectTab('repository')}
        >
          <FolderArchive size={18} />
          <span>Document Vault</span>
        </button>
      </nav>

      <div className="sidebar-footer">
        <button
          className="nav-item"
          onClick={onOpenSettings}
          style={{ width: '100%', justifyContent: 'flex-start' }}
        >
          <Settings size={18} />
          <span>API Connection</span>
        </button>
        <div style={{ padding: '0.75rem 0.85rem 0', fontSize: '0.72rem', color: 'var(--text-subtle)' }}>
          {engineInfo ? `${engineDisplayName} • ${deviceName}` : 'OCR Engine • Poppler'}
        </div>
      </div>
    </aside>
  );
};
