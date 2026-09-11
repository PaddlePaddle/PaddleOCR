/**
 * Document OCR Web Application - Types
 */

export type DocumentStatus = 'completed' | 'low_confidence' | 'warning' | 'failed' | 'error' | 'processing';

export type TextSource = 'embedded_pdf_text' | 'paddle_ocr' | 'rapid_ocr' | string;

export interface EngineInfo {
  engine: string;
  display_name: string;
  version: string;
  device: string;
  status: string;
  backend?: string;
  [key: string]: any;
}

export interface SupportedType {
  id: string;
  name: string;
  category: string;
}

export interface DashboardStats {
  total: number;
  ocr_processed: number;
  ocr_not_required: number;
  completed: number;
  failed: number;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_path: string;
  file_size: number;
  file_type: string;
  doc_type: string;
  document_type: string;
  ocr_required: boolean;
  text_source: TextSource;
  status: DocumentStatus;
  confidence: number;
  pages: number;
  reason?: string | null;
  extracted_fields: Record<string, any>;
  fields?: Record<string, any>;
  raw_fields?: Record<string, any>;
  field_confidences: Record<string, number>;
  extracted_text: string;
  checksum_valid?: boolean;
  checksum_reason?: string | null;
  cross_check?: {
    match: boolean;
    confidence: number;
    score: number;
    discrepancies: string[];
    matches?: string[];
  } | null;
  has_preview: boolean;
  preview_url: string | null;
  file_url: string;
  created_at: string;
}

export interface UploadProgress {
  step: 'idle' | 'uploading' | 'analyzing' | 'extracting' | 'verifying' | 'done' | 'error';
  percent: number;
  message: string;
}

export interface AuthConfig {
  baseUrl: string;
  authMode: 'none' | 'jwt' | 'api_key';
  token?: string;
  apiKey?: string;
  clientId?: string;
  clientSecret?: string;
}

export interface AuthStatusResponse {
  auth_enabled: boolean;
  auth_mode: string;
}

