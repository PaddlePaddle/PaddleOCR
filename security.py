"""
security.py
Authentication & Authorization for Company-Server OCR service:
- Supports short-lived JWT Bearer tokens with expiration
- Explicit client credential verification for /auth/token minting
- Swappable with static X-API-Key only via explicit opt-in (AUTH_MODE=dual or api_key)
- Strict fail-fast on startup if required secrets are unconfigured
"""

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from fastapi import HTTPException, Header, Query, status
import jwt

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "30"))

# Registered client credential store: client_id -> sha256_hex_hash
_registered_clients: Dict[str, str] = {}


def is_auth_enabled() -> bool:
    """
    Check if authentication is active.
    Authentication is DISABLED by default unless AUTH_ENABLED is explicitly set to 'true'.
    """
    val = os.getenv("AUTH_ENABLED", "").lower().strip()
    return val in ("true", "1", "yes")


def get_auth_mode() -> str:
    """Retrieve current AUTH_MODE. Returns 'disabled' unless AUTH_ENABLED=true."""
    if not is_auth_enabled():
        return "disabled"
    return os.getenv("AUTH_MODE", "jwt").lower().strip()


def get_jwt_secret() -> str:
    """Retrieve JWT secret from environment. Never returns a hardcoded default."""
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET environment variable is not set. A secure secret must be configured via secret management."
        )
    return secret


def get_static_api_key() -> Optional[str]:
    """Retrieve static API key if configured. Never returns a hardcoded default."""
    return os.getenv("API_KEY")


def validate_security_configuration():
    """
    Fail-fast check on service startup.
    Ensures that required secrets and client credentials are explicitly set when auth is enabled.
    In AUTH_MODE='disabled' (default), this check passes immediately with zero required secrets.
    """
    mode = get_auth_mode()
    if mode in ("disabled", "none", "off", "false"):
        return

    if mode in ("jwt", "dual"):
        jwt_sec = os.getenv("JWT_SECRET")
        if not jwt_sec or not jwt_sec.strip():
            raise RuntimeError(
                f"Missing required environment variable 'JWT_SECRET'. "
                f"In AUTH_MODE='{mode}', JWT_SECRET must be explicitly set via secret management."
            )
        # Load external client credentials if configured
        load_registered_clients()
        if len(_registered_clients) == 0:
            raise RuntimeError(
                f"No client credentials registered for /auth/token. "
                f"In AUTH_MODE='{mode}', client credentials must be configured via "
                f"REGISTERED_CLIENTS_JSON or REGISTERED_CLIENTS_FILE."
            )

    if mode in ("api_key", "dual"):
        api_key = os.getenv("API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError(
                f"Missing required environment variable 'API_KEY'. "
                f"In AUTH_MODE='{mode}', API_KEY must be explicitly set."
            )


# ==============================================================================
# Client Credential Management (Fix 1)
# ==============================================================================

def hash_client_secret(raw_secret: str) -> str:
    """Compute SHA-256 hex digest of a raw secret."""
    return hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()


def register_client(client_id: str, raw_secret: str):
    """Register a client with hashed secret in memory."""
    _registered_clients[client_id.strip()] = hash_client_secret(raw_secret)


def load_registered_clients():
    """Load registered clients from environment JSON string or file path."""
    # 1. From JSON string
    clients_json = os.getenv("REGISTERED_CLIENTS_JSON")
    if clients_json:
        try:
            data = json.loads(clients_json)
            if not isinstance(data, dict):
                raise ValueError("REGISTERED_CLIENTS_JSON must be a JSON object mapping client_id to secret.")
            for cid, sec_or_hash in data.items():
                val = str(sec_or_hash).strip()
                if len(val) == 64 and all(c in "0123456789abcdefABCDEF" for c in val):
                    _registered_clients[cid.strip()] = val.lower()
                else:
                    _registered_clients[cid.strip()] = hash_client_secret(val)
        except Exception as ex:
            raise RuntimeError(f"Failed to parse REGISTERED_CLIENTS_JSON: {ex}")

    # 2. From file path
    clients_file = os.getenv("REGISTERED_CLIENTS_FILE")
    if clients_file:
        if not os.path.exists(clients_file):
            raise RuntimeError(
                f"REGISTERED_CLIENTS_FILE is configured as '{clients_file}', but the file does not exist."
            )
        try:
            with open(clients_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("REGISTERED_CLIENTS_FILE content must be a JSON object mapping client_id to secret.")
            for cid, sec_or_hash in data.items():
                val = str(sec_or_hash).strip()
                if len(val) == 64 and all(c in "0123456789abcdefABCDEF" for c in val):
                    _registered_clients[cid.strip()] = val.lower()
                else:
                    _registered_clients[cid.strip()] = hash_client_secret(val)
        except Exception as ex:
            raise RuntimeError(f"Failed to load REGISTERED_CLIENTS_FILE '{clients_file}': {ex}")


def is_client_store_configured() -> bool:
    """Return True if any clients are currently registered in memory."""
    return len(_registered_clients) > 0


def verify_client_credentials(client_id: Optional[str], client_secret: Optional[str]) -> bool:
    """
    Verify client credentials against registered store.
    Constant-time comparison via hmac.compare_digest.
    """
    if not client_id or not client_secret:
        return False

    load_registered_clients()
    expected_hash = _registered_clients.get(client_id.strip())
    if not expected_hash:
        return False

    provided_hash = hash_client_secret(client_secret)
    return hmac.compare_digest(provided_hash, expected_hash)


def clear_registered_clients():
    """Helper to clear registered clients (for unit tests)."""
    _registered_clients.clear()


# ==============================================================================
# JWT Creation & Verification
# ==============================================================================

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed, short-lived JWT access token."""
    secret = get_jwt_secret()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=JWT_EXPIRATION_MINUTES))
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": "company-server-ocr",
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Validate a JWT token and ensure it has not expired."""
    secret = get_jwt_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def authenticate_request(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    token: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
) -> dict:
    """
    Authenticate incoming request using either JWT Bearer token (via Authorization
    header or ?token= query parameter) or static API Key (via X-API-Key header or
    ?api_key= query parameter).
    """
    mode = get_auth_mode()
    if mode in ("disabled", "none", "off", "false"):
        return {
            "sub": "anonymous",
            "auth_method": "none",
            "auth_mode": "disabled",
        }

    static_key = get_static_api_key()

    # 1. If static API Key mode strictly required
    if mode == "api_key":
        key_candidate = x_api_key or api_key
        if key_candidate and static_key and hmac.compare_digest(key_candidate, static_key):
            return {"sub": "api_key_client", "auth_method": "api_key"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )

    # 2. Check JWT Bearer token (from Authorization header or query parameter)
    jwt_candidate = None
    if authorization and authorization.startswith("Bearer "):
        jwt_candidate = authorization.split(" ", 1)[1].strip()
    elif token:
        jwt_candidate = token.strip()

    if jwt_candidate:
        payload = verify_jwt_token(jwt_candidate)
        payload["auth_method"] = "jwt"
        return payload

    # 3. Check static API Key (fallback only if AUTH_MODE == 'dual')
    key_candidate = x_api_key or api_key
    if mode == "dual" and key_candidate and static_key and hmac.compare_digest(key_candidate, static_key):
        return {"sub": "api_key_client", "auth_method": "api_key"}

    # If neither provided or matched
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid authentication credentials (Bearer JWT or X-API-Key required)",
        headers={"WWW-Authenticate": "Bearer"},
    )
