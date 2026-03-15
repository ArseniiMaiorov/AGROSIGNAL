from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from core.settings import get_settings


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(token: str) -> bytes:
    padding = "=" * ((4 - len(token) % 4) % 4)
    return base64.urlsafe_b64decode((token + padding).encode("ascii"))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = encoded.split("$", 2)
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=2**14,
        r=8,
        p=1,
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def encode_jwt(payload: dict[str, Any], *, secret: str | None = None) -> str:
    jwt_secret = secret or get_settings().AUTH_JWT_SECRET
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        (
            _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")),
            _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")),
        )
    )
    signature = hmac.new(jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_jwt(token: str, *, secret: str | None = None) -> dict[str, Any]:
    jwt_secret = secret or get_settings().AUTH_JWT_SECRET
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT structure")
    signing_input = ".".join(parts[:2])
    expected_sig = hmac.new(jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_decode(parts[2]), expected_sig):
        raise ValueError("Invalid JWT signature")
    payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    exp = int(payload.get("exp", 0) or 0)
    if exp <= int(time.time()):
        raise ValueError("JWT expired")
    return payload


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


@dataclass(slots=True)
class TokenIdentity:
    user_id: UUID
    organization_id: UUID
    email: str
    roles: tuple[str, ...]
    permissions: tuple[str, ...]


def issue_access_token(identity: TokenIdentity) -> str:
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": str(identity.user_id),
        "org": str(identity.organization_id),
        "email": identity.email,
        "roles": list(identity.roles),
        "permissions": list(identity.permissions),
        "type": "access",
        "iat": now,
        "exp": now + int(settings.AUTH_ACCESS_TTL_MINUTES) * 60,
    }
    return encode_jwt(payload)
