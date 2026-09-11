import type { DocumentItem, DashboardStats, SupportedType, AuthConfig, EngineInfo, AuthStatusResponse } from '../types';

const STORAGE_KEY_BASE_URL = 'ocr_app_base_url';
const SESSION_KEY_TOKEN = 'ocr_session_jwt_token';

export function getStoredBaseUrl(): string {
  try {
    if (typeof localStorage !== 'undefined') {
      const raw = localStorage.getItem(STORAGE_KEY_BASE_URL);
      if (raw) return raw;
    }
  } catch (e) {
    // Ignore storage read error
  }
  if (typeof window !== 'undefined' && window.location) {
    return window.location.port === '5173' ? 'http://localhost:8000' : window.location.origin;
  }
  return 'http://localhost:8000';
}

export function saveStoredBaseUrl(url: string): void {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_BASE_URL, url);
    }
  } catch (e) {
    // Ignore storage write error
  }
}

export class ApiService {
  private baseUrl: string;
  private token: string | null = null;
  private apiKey: string | null = null;
  private authEnabled: boolean = false;
  private authMode: string = 'disabled';
  private unauthorizedListeners: Array<() => void> = [];

  constructor() {
    this.baseUrl = getStoredBaseUrl();
    try {
      if (typeof sessionStorage !== 'undefined') {
        this.token = sessionStorage.getItem(SESSION_KEY_TOKEN);
      }
    } catch (e) {
      this.token = null;
    }
  }

  public setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, '');
    saveStoredBaseUrl(this.baseUrl);
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public setToken(token: string | null) {
    this.token = token;
    try {
      if (typeof sessionStorage !== 'undefined') {
        if (token) {
          sessionStorage.setItem(SESSION_KEY_TOKEN, token);
        } else {
          sessionStorage.removeItem(SESSION_KEY_TOKEN);
        }
      }
    } catch (e) {
      // Ignore session storage error
    }
  }

  public getToken(): string | null {
    return this.token;
  }

  public setApiKey(key: string | null) {
    this.apiKey = key;
  }

  public getApiKey(): string | null {
    return this.apiKey;
  }

  public async checkAuthStatus(): Promise<AuthStatusResponse> {
    try {
      const res = await fetch(this.getUrl('/api/auth-status'));
      if (res.ok) {
        const data: AuthStatusResponse = await res.json();
        this.authEnabled = Boolean(data.auth_enabled);
        this.authMode = data.auth_mode || 'disabled';
        return data;
      }
    } catch (e) {
      console.warn('Failed to fetch /api/auth-status, checking /health:', e);
    }

    // Fallback: check /health
    try {
      const health = await this.checkHealth();
      this.authEnabled = Boolean(health.auth_enabled);
      this.authMode = health.auth_mode || 'disabled';
      return { auth_enabled: this.authEnabled, auth_mode: this.authMode };
    } catch {
      this.authEnabled = false;
      this.authMode = 'disabled';
      return { auth_enabled: false, auth_mode: 'disabled' };
    }
  }

  public isAuthEnabled(): boolean {
    return this.authEnabled;
  }

  public setAuthEnabled(enabled: boolean, mode: string = 'jwt') {
    this.authEnabled = enabled;
    this.authMode = mode;
  }

  public isAuthenticated(): boolean {
    if (!this.authEnabled) {
      return true;
    }
    return Boolean(this.token || this.apiKey);
  }

  public onUnauthorized(listener: () => void): () => void {
    this.unauthorizedListeners.push(listener);
    return () => {
      this.unauthorizedListeners = this.unauthorizedListeners.filter((l) => l !== listener);
    };
  }

  private notifyUnauthorized() {
    this.setToken(null);
    for (const listener of this.unauthorizedListeners) {
      try {
        listener();
      } catch (e) {
        console.error('Error in unauthorized listener:', e);
      }
    }
  }

  public getConfig(): AuthConfig {
    return {
      baseUrl: this.baseUrl,
      authMode: this.apiKey ? 'api_key' : 'jwt',
      token: this.token || undefined,
      apiKey: this.apiKey || undefined,
    };
  }

  public updateConfig(newConfig: Partial<AuthConfig>) {
    if (newConfig.baseUrl) this.setBaseUrl(newConfig.baseUrl);
    if (newConfig.token !== undefined) this.setToken(newConfig.token || null);
    if (newConfig.apiKey !== undefined) this.setApiKey(newConfig.apiKey || null);
  }

  private getHeaders(isFormData = false): Record<string, string> {
    const headers: Record<string, string> = {};
    if (!isFormData) {
      headers['Content-Type'] = 'application/json';
    }
    // Only attach auth headers if backend reports auth is enabled and a credential exists
    if (this.authEnabled) {
      if (this.token) {
        headers['Authorization'] = `Bearer ${this.token}`;
      } else if (this.apiKey) {
        headers['X-API-Key'] = this.apiKey;
      }
    }
    return headers;
  }

  private getUrl(path: string): string {
    const base = this.baseUrl.replace(/\/+$/, '');
    return `${base}${path.startsWith('/') ? path : '/' + path}`;
  }

  private async fetchWithAuth(url: string, options: RequestInit = {}): Promise<Response> {
    const res = await fetch(url, options);
    if (res.status === 401) {
      this.notifyUnauthorized();
      throw new Error('Authentication required or session expired (HTTP 401). Please sign in.');
    }
    return res;
  }

  public async checkHealth(): Promise<{ status: string; version: string; auth_enabled?: boolean; auth_mode?: string }> {
    const res = await fetch(this.getUrl('/health'));
    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }
    return res.json();
  }

  public async getEngineInfo(): Promise<EngineInfo> {
    const res = await fetch(this.getUrl('/engine-info'));
    if (!res.ok) {
      throw new Error(`Failed to load engine info (HTTP ${res.status})`);
    }
    const data = await res.json();
    const active = data.active_engine || data.engine || 'none';
    const display = data.display_name || (
      active === 'rapidocr' ? 'RapidOCR' :
      active === 'paddleocr' ? 'PaddleOCR' :
      active !== 'none' ? active : 'OCR'
    );
    return {
      engine: active,
      display_name: display,
      version: data.version || data.rapidocr_version || data.paddleocr_version || '1.0',
      device: data.device || 'CPU',
      status: data.status || 'ready',
      backend: data.backend || active,
      ...data,
    };
  }

  public async login(clientId: string, clientSecret: string): Promise<string> {
    const formData = new URLSearchParams();
    formData.append('client_id', clientId.trim());
    formData.append('client_secret', clientSecret.trim());

    const res = await fetch(this.getUrl('/auth/token'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString(),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Authentication failed' }));
      throw new Error(err.detail || `Login failed with status ${res.status}`);
    }

    const data = await res.json();
    this.setToken(data.access_token);
    return data.access_token;
  }

  public async mintToken(clientId: string, clientSecret: string): Promise<string> {
    return this.login(clientId, clientSecret);
  }

  public logout(): void {
    this.setToken(null);
    this.notifyUnauthorized();
  }

  public async getSupportedTypes(): Promise<SupportedType[]> {
    const res = await fetch(this.getUrl('/api/supported-types'));
    if (!res.ok) throw new Error('Failed to load supported document types');
    return res.json();
  }

  public async getStats(): Promise<DashboardStats> {
    const res = await this.fetchWithAuth(this.getUrl('/api/stats'), {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error('Failed to load dashboard statistics');
    return res.json();
  }

  public async getDocuments(params?: {
    search?: string;
    docType?: string;
    ocrRequired?: boolean;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ items: DocumentItem[]; total: number }> {
    const query = new URLSearchParams();
    if (params?.search) query.set('search', params.search);
    if (params?.docType && params.docType !== 'all') query.set('doc_type', params.docType);
    if (params?.ocrRequired !== undefined) query.set('ocr_required', String(params.ocrRequired));
    if (params?.status && params.status !== 'all') query.set('status', params.status);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));

    const url = this.getUrl(`/api/documents?${query.toString()}`);
    const res = await this.fetchWithAuth(url, { headers: this.getHeaders() });
    if (!res.ok) throw new Error('Failed to load documents');
    return res.json();
  }

  public async getDocument(docId: string): Promise<DocumentItem> {
    const res = await this.fetchWithAuth(this.getUrl(`/api/documents/${docId}`), {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error('Document not found');
    return res.json();
  }

  public async deleteDocument(docId: string): Promise<boolean> {
    const res = await this.fetchWithAuth(this.getUrl(`/api/documents/${docId}`), {
      method: 'DELETE',
      headers: this.getHeaders(),
    });
    return res.ok;
  }

  public async uploadDocument(
    file: File,
    docType?: string,
    expectedData?: string,
    onProgress?: (percent: number) => void
  ): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append('file', file);
    if (docType && docType !== 'auto') {
      formData.append('doc_type', docType);
    }
    if (expectedData) {
      formData.append('expected_data', expectedData);
    }

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.getUrl('/api/upload'));

      if (this.authEnabled) {
        if (this.token) {
          xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
        } else if (this.apiKey) {
          xhr.setRequestHeader('X-API-Key', this.apiKey);
        }
      }

      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable && onProgress) {
          const pct = Math.round((evt.loaded / evt.total) * 60);
          onProgress(pct);
        }
      };

      xhr.onload = () => {
        if (xhr.status === 401) {
          this.notifyUnauthorized();
          reject(new Error('Session expired or unauthorized. Please sign in again.'));
          return;
        }

        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            if (onProgress) onProgress(100);
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch (e) {
            reject(new Error('Invalid JSON returned by server'));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || `Upload failed: ${xhr.statusText}`));
          } catch {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => {
        reject(new Error('Network error during upload'));
      };

      xhr.send(formData);
    });
  }

  public getFileUrl(docId: string, download = false): string {
    const params = new URLSearchParams();
    if (download) params.set('download', 'true');
    if (this.authEnabled) {
      if (this.token) params.set('token', this.token);
      else if (this.apiKey) params.set('api_key', this.apiKey);
    }
    const qs = params.toString();
    return this.getUrl(`/api/documents/${docId}/file${qs ? '?' + qs : ''}`);
  }

  public getPreviewUrl(docId: string): string {
    if (this.authEnabled) {
      const params = new URLSearchParams();
      if (this.token) params.set('token', this.token);
      else if (this.apiKey) params.set('api_key', this.apiKey);
      const qs = params.toString();
      return this.getUrl(`/api/documents/${docId}/preview${qs ? '?' + qs : ''}`);
    }
    return this.getUrl(`/api/documents/${docId}/preview`);
  }
}

export const api = new ApiService();
