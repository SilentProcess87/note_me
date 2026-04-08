"""
Encrypted secrets store using Windows DPAPI (Data Protection API).

API keys are encrypted with the current Windows user's credentials.
Even if someone decompiles the binary and reads the config file,
they only see ciphertext that cannot be decrypted without the
original user's Windows login session.

On non-Windows platforms, falls back to base64 obfuscation
(not secure, but prevents casual inspection).
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import logging
import sys
from typing import Optional

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Windows DPAPI via ctypes (no pywin32 dependency)
# ------------------------------------------------------------------ #

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_encrypt(plaintext: bytes) -> bytes:
    """Encrypt bytes using CryptProtectData (current user scope)."""
    blob_in = _DATA_BLOB(len(plaintext), ctypes.create_string_buffer(plaintext, len(plaintext)))
    blob_out = _DATA_BLOB()

    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        "NoteMe",         # description (stored with ciphertext)
        None,             # optional entropy
        None,             # reserved
        None,             # prompt struct
        0x01,             # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(blob_out),
    ):
        raise OSError("CryptProtectData failed")

    encrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return encrypted


def _dpapi_decrypt(ciphertext: bytes) -> bytes:
    """Decrypt bytes using CryptUnprotectData (current user scope)."""
    blob_in = _DATA_BLOB(len(ciphertext), ctypes.create_string_buffer(ciphertext, len(ciphertext)))
    blob_out = _DATA_BLOB()

    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,             # description out
        None,             # entropy
        None,             # reserved
        None,             # prompt struct
        0x01,             # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(blob_out),
    ):
        raise OSError("CryptUnprotectData failed")

    decrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return decrypted


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def encrypt_key(api_key: str) -> str:
    """Encrypt an API key and return a base64-encoded string safe for config files."""
    if not api_key:
        return ""
    try:
        if sys.platform == "win32":
            encrypted = _dpapi_encrypt(api_key.encode("utf-8"))
            return "DPAPI:" + base64.b64encode(encrypted).decode("ascii")
        else:
            # Fallback: base64 obfuscation (not truly secure)
            return "B64:" + base64.b64encode(api_key.encode("utf-8")).decode("ascii")
    except Exception as exc:
        log.warning("Failed to encrypt API key: %s", exc)
        return "B64:" + base64.b64encode(api_key.encode("utf-8")).decode("ascii")


def decrypt_key(stored: str) -> str:
    """Decrypt a stored API key string back to plaintext."""
    if not stored:
        return ""
    try:
        if stored.startswith("DPAPI:"):
            ciphertext = base64.b64decode(stored[6:])
            return _dpapi_decrypt(ciphertext).decode("utf-8")
        elif stored.startswith("B64:"):
            return base64.b64decode(stored[4:]).decode("utf-8")
        else:
            # Legacy: treat as plaintext (first run before encryption)
            return stored
    except Exception as exc:
        log.warning("Failed to decrypt API key: %s", exc)
        return ""


def is_encrypted(stored: str) -> bool:
    """Check if a stored key is already encrypted."""
    return stored.startswith("DPAPI:") or stored.startswith("B64:")
