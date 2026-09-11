import React from 'react';
import { Settings, RefreshCw, LogOut } from 'lucide-react';

interface HeaderProps {
  title: string;
  isBackendConnected: boolean;
  onOpenSettings: () => void;
  onRefresh: () => void;
  onLogout?: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  isBackendConnected,
  onOpenSettings,
  onRefresh,
  onLogout,
  isRefreshing = false,
}) => {
  return (
    <header className="top-header">
      <div className="header-title-box">
        <h1 className="page-title">{title}</h1>
      </div>

      <div className="header-actions">
        <div className={`status-pill ${isBackendConnected ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {isBackendConnected ? 'Backend Connected' : 'Server Disconnected'}
        </div>

        <button
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh Data"
          style={{ padding: '0.45rem 0.75rem' }}
        >
          <RefreshCw size={15} className={isRefreshing ? 'spin-anim' : ''} />
          <span style={{ fontSize: '0.8rem' }}>Sync</span>
        </button>

        <button
          className="btn btn-secondary"
          onClick={onOpenSettings}
          style={{ padding: '0.45rem 0.75rem' }}
          title="Server Settings"
        >
          <Settings size={15} />
          <span style={{ fontSize: '0.8rem' }}>Settings</span>
        </button>

        {onLogout && (
          <button
            className="btn btn-secondary"
            onClick={onLogout}
            style={{ padding: '0.45rem 0.75rem', borderColor: 'var(--border-subtle)' }}
            title="Sign Out"
          >
            <LogOut size={15} color="var(--accent-rose)" />
            <span style={{ fontSize: '0.8rem' }}>Sign Out</span>
          </button>
        )}
      </div>
    </header>
  );
};
