import React, { useState, useEffect, useCallback } from 'react';
import {
  FileText,
  FileCode,
  ShieldCheck,
  ArrowLeft,
  Share2,
} from 'lucide-react';

import type { DocumentItem, SupportedType, DashboardStats as StatsType, EngineInfo } from './types';
import { api } from './services/api';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import type { NavTab } from './components/Sidebar';
import { DashboardStats } from './components/DashboardStats';
import { UploadCard } from './components/UploadCard';
import { OcrDecisionBadge } from './components/OcrDecisionBadge';
import { DocumentPreview } from './components/DocumentPreview';
import { ExtractedFields } from './components/ExtractedFields';
import { ExtractedTextViewer } from './components/ExtractedTextViewer';
import { JsonResultViewer } from './components/JsonResultViewer';
import { DocumentsTable } from './components/DocumentsTable';
import { SettingsModal } from './components/SettingsModal';
import { LoginView } from './components/LoginView';
import { Toast } from './components/Toast';
import type { ToastMessage } from './components/Toast';

type InspectTab = 'fields' | 'text' | 'json';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
  const [inspectTab, setInspectTab] = useState<InspectTab>('fields');
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);

  const [supportedTypes, setSupportedTypes] = useState<SupportedType[]>([]);
  const [engineInfo, setEngineInfo] = useState<EngineInfo | null>(null);
  const [stats, setStats] = useState<StatsType>({
    total: 0,
    ocr_processed: 0,
    ocr_not_required: 0,
    completed: 0,
    failed: 0,
  });
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [authEnabled, setAuthEnabled] = useState<boolean>(false);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(api.isAuthenticated());
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = (message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const id = String(Date.now());
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const loadData = useCallback(async () => {
    setIsRefreshing(true);
    try {
      // 1. Check health & dynamic backend auth status
      await api.checkHealth();
      setIsBackendConnected(true);

      const [authStatus, typesData, engineData] = await Promise.all([
        api.checkAuthStatus().catch(() => ({ auth_enabled: false, auth_mode: 'disabled' })),
        api.getSupportedTypes().catch(() => []),
        api.getEngineInfo().catch(() => null),
      ]);

      const isEnabled = Boolean(authStatus.auth_enabled);
      setAuthEnabled(isEnabled);
      const authed = api.isAuthenticated();
      setIsAuthenticated(authed);

      setSupportedTypes(typesData);
      if (engineData) setEngineInfo(engineData);

      // Only fetch protected data if not gated by enabled auth
      if (isEnabled && !authed) {
        return;
      }

      // 2. Fetch protected stats and documents in parallel
      const [statsData, docsData] = await Promise.all([
        api.getStats().catch(() => ({
          total: 0,
          ocr_processed: 0,
          ocr_not_required: 0,
          completed: 0,
          failed: 0,
        })),
        api.getDocuments({ limit: 100 }).catch(() => ({ items: [], total: 0 })),
      ]);

      setStats(statsData);
      setDocuments(docsData.items);
    } catch (err: any) {
      setIsBackendConnected(false);
      addToast('Cannot connect to FastAPI OCR server. Check settings or start backend.', 'error');
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const unsubscribeAuth = api.onUnauthorized(() => {
      if (api.isAuthEnabled()) {
        setIsAuthenticated(false);
        addToast('Session expired or unauthorized. Please sign in again.', 'info');
      }
    });

    loadData();

    // Periodic health poll every 25 seconds
    const interval = setInterval(async () => {
      try {
        await api.checkHealth();
        setIsBackendConnected(true);
      } catch {
        setIsBackendConnected(false);
      }
    }, 25000);

    return () => {
      unsubscribeAuth();
      clearInterval(interval);
    };
  }, [loadData]);

  const handleDocumentUploaded = (doc: DocumentItem) => {
    setSelectedDoc(doc);
    setInspectTab('fields');
    addToast(`Successfully processed "${doc.filename}"!`, 'success');
    loadData();
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      const ok = await api.deleteDocument(id);
      if (ok) {
        addToast('Document deleted from vault', 'success');
        if (selectedDoc?.id === id) {
          setSelectedDoc(null);
        }
        loadData();
      }
    } catch (err: any) {
      addToast(`Delete failed: ${err.message}`, 'error');
    }
  };

  const getPageTitle = () => {
    if (selectedDoc) {
      return `Document Inspection: ${selectedDoc.filename}`;
    }
    switch (currentTab) {
      case 'dashboard':
        return 'Document OCR Dashboard';
      case 'upload':
        return 'Upload & Ingest Document';
      case 'repository':
        return 'Document Vault Repository';
      default:
        return 'DocuScan AI';
    }
  };

  // Only gate the dashboard if the backend actually reports auth is enabled AND user has no valid token
  if (authEnabled && !isAuthenticated) {
    return (
      <>
        <LoginView
          onLoginSuccess={() => {
            setIsAuthenticated(true);
            loadData();
          }}
        />
        <Toast toasts={toasts} onDismiss={removeToast} />
      </>
    );
  }

  return (
    <div className="app-layout">
      <Sidebar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          setCurrentTab(tab);
          setSelectedDoc(null);
        }}
        onOpenSettings={() => setIsSettingsOpen(true)}
        engineInfo={engineInfo}
      />

      <div className="main-content">
        <Header
          title={getPageTitle()}
          isBackendConnected={isBackendConnected}
          onOpenSettings={() => setIsSettingsOpen(true)}
          onRefresh={loadData}
          onLogout={
            authEnabled
              ? () => {
                  api.logout();
                  setIsAuthenticated(false);
                  addToast('Signed out successfully.', 'info');
                }
              : undefined
          }
          isRefreshing={isRefreshing}
        />

        <main className="content-body">
          {/* If inspecting a specific document */}
          {selectedDoc ? (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setSelectedDoc(null)}
                  style={{ padding: '0.45rem 0.85rem' }}
                >
                  <ArrowLeft size={16} />
                  <span>Back to {currentTab === 'upload' ? 'Upload' : 'Repository'}</span>
                </button>

                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      navigator.clipboard.writeText(window.location.href);
                      addToast('Inspection link copied', 'info');
                    }}
                    style={{ padding: '0.45rem 0.85rem' }}
                  >
                    <Share2 size={15} />
                    <span>Share</span>
                  </button>

                  <button
                    className="btn btn-danger"
                    onClick={() => {
                      if (window.confirm(`Delete "${selectedDoc.filename}" from vault?`)) {
                        handleDeleteDocument(selectedDoc.id);
                      }
                    }}
                    style={{ padding: '0.45rem 0.85rem' }}
                  >
                    <span>Delete</span>
                  </button>
                </div>
              </div>

              {/* Decision Banner */}
              <OcrDecisionBadge document={selectedDoc} engineInfo={engineInfo} />

              {/* 2-Column Inspection Grid */}
              <div className="result-grid">
                {/* Left Column: Visual Document Preview */}
                <DocumentPreview document={selectedDoc} />

                {/* Right Column: Tabbed Inspector */}
                <div className="inspect-panel">
                  <div className="tabs-nav">
                    <button
                      className={`tab-btn ${inspectTab === 'fields' ? 'active' : ''}`}
                      onClick={() => setInspectTab('fields')}
                    >
                      <ShieldCheck size={16} />
                      <span>Extracted Fields</span>
                    </button>

                    <button
                      className={`tab-btn ${inspectTab === 'text' ? 'active' : ''}`}
                      onClick={() => setInspectTab('text')}
                    >
                      <FileText size={16} />
                      <span>Extracted Text ({selectedDoc.extracted_text.length} chars)</span>
                    </button>

                    <button
                      className={`tab-btn ${inspectTab === 'json' ? 'active' : ''}`}
                      onClick={() => setInspectTab('json')}
                    >
                      <FileCode size={16} />
                      <span>Raw JSON Payload</span>
                    </button>
                  </div>

                  <div className="tab-content">
                    {inspectTab === 'fields' && (
                      <ExtractedFields document={selectedDoc} onCopyToast={addToast} />
                    )}

                    {inspectTab === 'text' && (
                      <ExtractedTextViewer
                        text={selectedDoc.extracted_text}
                        filename={selectedDoc.filename}
                        onCopyToast={addToast}
                      />
                    )}

                    {inspectTab === 'json' && (
                      <JsonResultViewer document={selectedDoc} onCopyToast={addToast} />
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Dashboard Tab */}
              {currentTab === 'dashboard' && (
                <>
                  <DashboardStats
                    stats={stats}
                    engineInfo={engineInfo}
                    documents={documents}
                    onNavigateTab={(tab) => setCurrentTab(tab)}
                  />

                  <UploadCard
                    supportedTypes={supportedTypes}
                    onUploadSuccess={handleDocumentUploaded}
                    onError={(msg) => addToast(msg, 'error')}
                  />

                  <div style={{ marginTop: '2.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Recent Document Ingestions</h2>
                      <button
                        className="btn btn-ghost"
                        onClick={() => setCurrentTab('repository')}
                        style={{ fontSize: '0.82rem' }}
                      >
                        View All in Vault →
                      </button>
                    </div>

                    <DocumentsTable
                      documents={documents.slice(0, 5)}
                      supportedTypes={supportedTypes}
                      onSelectDocument={(doc) => setSelectedDoc(doc)}
                      onDeleteDocument={handleDeleteDocument}
                      onRefresh={loadData}
                      isLoading={isRefreshing}
                      engineInfo={engineInfo}
                    />
                  </div>
                </>
              )}

              {/* Upload Tab */}
              {currentTab === 'upload' && (
                <div>
                  <UploadCard
                    supportedTypes={supportedTypes}
                    onUploadSuccess={handleDocumentUploaded}
                    onError={(msg) => addToast(msg, 'error')}
                  />
                </div>
              )}

              {/* Document Repository Tab */}
              {currentTab === 'repository' && (
                <div>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                      Stored Document Vault
                    </h2>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Search, inspect, and retrieve previously processed documents with full verification metrics.
                    </p>
                  </div>

                  <DocumentsTable
                    documents={documents}
                    supportedTypes={supportedTypes}
                    onSelectDocument={(doc) => setSelectedDoc(doc)}
                    onDeleteDocument={handleDeleteDocument}
                    onRefresh={loadData}
                    isLoading={isRefreshing}
                    engineInfo={engineInfo}
                  />
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSaved={() => {
          addToast('Settings saved. Refreshing data...', 'info');
          loadData();
        }}
      />

      {/* Toast Notifications */}
      <Toast toasts={toasts} onDismiss={removeToast} />
    </div>
  );
};

export default App;

