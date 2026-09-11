import React, { useState } from 'react';
import { X, Check, Server, Key, Shield, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, onSaved }) => {
  const currentConfig = api.getConfig();
  const [baseUrl, setBaseUrl] = useState(currentConfig.baseUrl);
  const [clientId, setClientId] = useState(currentConfig.clientId || '');
  const [clientSecret, setClientSecret] = useState(currentConfig.clientSecret || '');
  const [bearerToken, setBearerToken] = useState(currentConfig.token || '');
  const [apiKey, setApiKey] = useState(currentConfig.apiKey || '');
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    const start = performance.now();
    try {
      api.updateConfig({ baseUrl, token: bearerToken || undefined, apiKey: apiKey || undefined });
      const health = await api.checkHealth();
      const latency = Math.round(performance.now() - start);
      setTestResult({
        ok: true,
        message: `Connected successfully! Latency: ${latency}ms (Service v${health.version})`,
      });
    } catch (err: any) {
      setTestResult({
        ok: false,
        message: `Connection failed: ${err.message}`,
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleMintToken = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      api.updateConfig({ baseUrl });
      const token = await api.mintToken(clientId, clientSecret);
      setBearerToken(token);
      setTestResult({
        ok: true,
        message: 'Successfully minted and saved JWT access token!',
      });
    } catch (err: any) {
      setTestResult({
        ok: false,
        message: `Failed to mint token: ${err.message}`,
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = () => {
    api.updateConfig({
      baseUrl,
      clientId,
      clientSecret,
      token: bearerToken || undefined,
      apiKey: apiKey || undefined,
    });
    onSaved();
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Server size={18} color="#818cf8" />
            <h2 className="modal-title">API Connection & Auth Settings</h2>
          </div>
          <button className="copy-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label className="form-label">FastAPI Base URL</label>
            <input
              type="text"
              className="form-input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>

          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Shield size={16} color="#34d399" />
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Client Credentials (POST /auth/token)</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <div className="form-group">
                <label className="form-label">Client ID</label>
                <input
                  type="text"
                  className="form-input"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                  placeholder="Enter client ID"
                />
              </div>
              <div className="form-group">
                <label className="form-label">Client Secret</label>
                <input
                  type="password"
                  className="form-input"
                  value={clientSecret}
                  onChange={(e) => setClientSecret(e.target.value)}
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleMintToken}
              disabled={isTesting || !clientId || !clientSecret}
              style={{ width: '100%', marginBottom: '1rem' }}
            >
              <Key size={14} />
              <span>Mint JWT Access Token</span>
            </button>
          </div>

          <div className="form-group">
            <label className="form-label">Direct Bearer Token (Optional)</label>
            <input
              type="password"
              className="form-input"
              value={bearerToken}
              onChange={(e) => setBearerToken(e.target.value)}
              placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            />
          </div>

          <div className="form-group">
            <label className="form-label">Static X-API-Key (Optional)</label>
            <input
              type="password"
              className="form-input"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter static API key if configured"
            />
          </div>

          {testResult && (
            <div
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: testResult.ok ? 'var(--success-bg)' : 'var(--error-bg)',
                border: `1px solid ${testResult.ok ? 'var(--success-border)' : 'var(--error-border)'}`,
                color: testResult.ok ? '#34d399' : '#f87171',
                fontSize: '0.82rem',
              }}
            >
              {testResult.message}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleTestConnection}
            disabled={isTesting}
          >
            <RefreshCw size={14} className={isTesting ? 'spin-anim' : ''} />
            <span>Test Ping</span>
          </button>

          <button type="button" className="btn btn-primary" onClick={handleSave}>
            <Check size={15} />
            <span>Save Configuration</span>
          </button>
        </div>
      </div>
    </div>
  );
};
