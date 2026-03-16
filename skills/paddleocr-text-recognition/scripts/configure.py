#!/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration Wizard for PaddleOCR Text Recognition

Supports two modes:
1. Interactive mode (default): python configure.py
2. CLI mode: python configure.py --api-url URL --token TOKEN

Supports pasting Python code format (e.g., PADDLEOCR_OCR_API_URL = "...").


Get your API credentials at: https://paddleocr.com
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_input(user_input: str) -> str:
    """
    Intelligently parse user input, supporting multiple formats:
    - API_URL = "https://..."
    - "https://..."
    - https://...
    - TOKEN = "abc123..."

    Returns the extracted value
    """
    user_input = user_input.strip()

    # Format 1: KEY = "value" or KEY = 'value'
    match = re.match(r'^\w+\s*=\s*["\'](.+?)["\']$', user_input)
    if match:
        return match.group(1)

    # Format 2: "value" or 'value'
    match = re.match(r'^["\'](.+?)["\']$', user_input)
    if match:
        return match.group(1)

    # Format 3: value (direct input)
    return user_input


def validate_api_url(url: str) -> str:
    """
    Validate and normalize OCR API endpoint URL.
    Requires a full endpoint ending with /ocr.
    """
    url = url.strip()
    if not url:
        raise ValueError("API URL cannot be empty")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    url = url.rstrip("/")

    api_path = urlparse(url).path.rstrip("/")
    if not api_path.endswith("/ocr"):
        raise ValueError(
            "API URL must be a full endpoint ending with /ocr. "
            "Example: https://your-service.paddleocr.com/ocr"
        )

    return url


def mask_token(token: str) -> str:
    """Mask token, only show first and last parts"""
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


def test_connection(api_url: str, token: str) -> bool:
    """Test API connection (optional)"""
    try:
        import httpx

        print("\nTesting connection...")

        # Simple test request (using a small base64 image)
        test_payload = {
            "file": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "fileType": 1,
            "visualize": False,
        }

        headers = {
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
        }

        client = httpx.Client(timeout=10.0)
        try:
            resp = client.post(api_url, json=test_payload, headers=headers)
            resp_json = resp.json()

            if resp.status_code == 200 and resp_json.get("errorCode") == 0:
                print("[OK] API connection successful!")
                print("[OK] OCR function is working!")
                return True
            elif resp.status_code == 403:
                print(
                    "[FAIL] Token verification failed, please check if the Token is correct"
                )
                return False
            elif resp.status_code == 429:
                print("[WARN] API quota exhausted, but connection is working")
                return True
            else:
                print(
                    f"[WARN] API returned error: {resp_json.get('errorMsg', 'Unknown error')}"
                )
                return False
        finally:
            client.close()

    except ImportError:
        print("[WARN] httpx not installed, skipping connection test")
        print("   Install with: pip install httpx")
        return True
    except Exception as e:
        print(f"[FAIL] Connection test failed: {e}")
        return False


def save_config(
    api_url: str, token: str, project_root: Path, quiet: bool = False
) -> bool:
    """
    Save configuration to .env file

    Args:
        api_url: Normalized API URL
        token: Access token
        project_root: Project root directory
        quiet: If True, suppress output messages

    Returns:
        True if successful, False otherwise
    """
    env_file = project_root / ".env"

    # Read existing configuration (if exists)
    existing_config = {}
    if env_file.exists():
        if not quiet:
            print(f"\nDetected existing configuration file: {env_file}")
            overwrite = input("Overwrite? [Y/n]: ").strip().lower()
            if overwrite == "n":
                print("Configuration cancelled")
                return False

        # Preserve other configuration items (excluding old and new text-recognition keys)
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        # Skip old and new text-recognition keys (will be overwritten)
                        if key not in [
                            "API_URL",
                            "PADDLE_OCR_TOKEN",
                            "AISTUDIO_HOST",
                            "PADDLEOCR_OCR_API_URL",
                            "PADDLEOCR_ACCESS_TOKEN",
                        ]:
                            existing_config[key] = value.strip()

    # Write new configuration
    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("# PaddleOCR Skills Configuration\n")
            f.write("# This file was automatically generated by configure.py\n")
            f.write("# Get your API credentials at: https://paddleocr.com\n\n")

            f.write("# PaddleOCR Text Recognition Configuration\n")
            f.write(f"PADDLEOCR_OCR_API_URL={api_url}\n")
            f.write(f"PADDLEOCR_ACCESS_TOKEN={token}\n")

            # Write other preserved configurations
            if existing_config:
                f.write("\n# Other configurations\n")
                for key, value in existing_config.items():
                    f.write(f"{key}={value}\n")

        if not quiet:
            print(f"\n[OK] Configuration saved to: {env_file}")
        return True

    except Exception as e:
        print(f"\n[FAIL] Failed to save configuration: {e}")
        return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="PaddleOCR Text Recognition Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python configure.py

  # CLI mode (non-interactive)
  python configure.py --api-url "https://xxx.aistudio-app.com/ocr" --token "your_token"
        """,
    )
    parser.add_argument("--api-url", help="API URL (non-interactive mode)")
    parser.add_argument("--token", help="Access token (non-interactive mode)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output messages")

    args = parser.parse_args()

    # Get project root directory (shared .env location, consistent with lib.py)
    project_root = Path(__file__).parent.parent.parent

    # ========================================
    # CLI Mode (non-interactive)
    # ========================================
    if args.api_url and args.token:
        try:
            # Validate full OCR endpoint URL
            api_url = validate_api_url(parse_input(args.api_url))
            token = parse_input(args.token)

            # Validate
            if len(token) < 16:
                print("Error: Token seems too short. Please check and try again.")
                sys.exit(1)

            # Save configuration (CLI mode always overwrites without asking)
            if save_config(api_url, token, project_root, quiet=True):
                if not args.quiet:
                    print("\n[OK] Configuration complete!")
                    print(f"  PADDLEOCR_OCR_API_URL: {api_url}")
                    print(f"  PADDLEOCR_ACCESS_TOKEN: {mask_token(token)}")
                sys.exit(0)
            else:
                sys.exit(1)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.api_url or args.token:
        print("Error: Both --api-url and --token are required for CLI mode")
        print("Run without arguments for interactive mode")
        sys.exit(1)

    # ========================================
    # Interactive Mode
    # ========================================
    print("\n" + "=" * 60)
    print("PaddleOCR Text Recognition - Configuration Wizard")
    print("=" * 60)
    print("\nGet your API credentials at: https://paddleocr.com\n")

    # ========================================
    # Step 1: Get API URL
    # ========================================
    print("[Step 1/2] Please enter your API URL")
    print("Tip: You can paste directly, for example:")
    print('  PADDLEOCR_OCR_API_URL = "https://your-service.paddleocr.com/ocr"')
    print("  or: https://your-service.paddleocr.com/ocr")
    print()

    while True:
        api_url_input = input("> ").strip()

        if not api_url_input:
            print("Error: API URL cannot be empty, please enter again")
            continue

        # Parse input
        api_url_raw = parse_input(api_url_input)

        # Validate full OCR endpoint URL
        try:
            api_url = validate_api_url(api_url_raw)
            print(f"[OK] Recognized: {api_url}\n")
            break
        except Exception as e:
            print(f"Error: Cannot parse API URL: {e}")
            print("Please enter again\n")

    # ========================================
    # Step 2: Get Token
    # ========================================
    print("[Step 2/2] Please enter your Access Token")
    print("Tip: You can paste directly, for example:")
    print('  PADDLEOCR_ACCESS_TOKEN = "your_token_here"')
    print("  or: your_token_here")
    print()

    while True:
        token_input = input("> ").strip()

        if not token_input:
            print("Error: Token cannot be empty, please enter again")
            continue

        # Parse input
        token = parse_input(token_input)

        if len(token) < 16:
            print("[WARN] Token length seems too short, please confirm if correct")
            confirm = input("Continue? [y/N]: ").strip().lower()
            if confirm != "y":
                continue

        print(f"[OK] Recognized: {mask_token(token)}\n")
        break

    # ========================================
    # Save configuration
    # ========================================
    print("=" * 60)
    print("Saving configuration...")
    print("=" * 60)

    if not save_config(api_url, token, project_root):
        sys.exit(1)

    # ========================================
    # Test connection (optional)
    # ========================================
    print("\n" + "=" * 60)
    test_choice = input("Test connection? [Y/n]: ").strip().lower()

    if test_choice != "n":
        success = test_connection(api_url, token)
        if not success:
            print("\n[WARN] Connection test failed, but configuration has been saved")
            print("  Please check if API URL and Token are correct")

    # ========================================
    # Complete
    # ========================================
    print("\n" + "=" * 60)
    print("Configuration complete!")
    print("=" * 60)
    print("\nYou can now use the OCR function:")
    print(f"  cd {project_root}")
    print('  python scripts/ocr_caller.py --file-url "https://example.com/image.jpg"')
    print("\nTo reconfigure, run this script again.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
