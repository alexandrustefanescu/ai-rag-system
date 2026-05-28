"""Clerk JWT verification with JWKS caching."""

import time

import httpx
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from jwt.algorithms import RSAAlgorithm
from jwt.api_jwt import PyJWT
from jwt.exceptions import DecodeError, InvalidSignatureError

from rag_system.config import ClerkConfig

_jwks_cache: dict[str, tuple[str, float]] = {}
_cache_ttl = 300  # 5 minutes


def _get_jwks_url(domain: str) -> str:
    """Build the JWKS URL for a Clerk frontend domain."""
    domain = domain.rstrip("/")
    return f"{domain}/.well-known/jwks.json"


def _fetch_jwks(jwks_url: str) -> list[dict]:
    """Fetch JWKS from Clerk and return the keys list."""
    response = httpx.get(jwks_url, timeout=10)
    response.raise_for_status()
    return response.json().get("keys", [])


def _get_public_key(token: str, jwks_url: str) -> RSAPublicKey:
    """Retrieve the RSA public key matching the token's 'kid' header."""
    unverified_header = PyJWT().get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise InvalidSignatureError("Missing 'kid' in token header")

    cached = _jwks_cache.get(kid)
    if cached:
        pem, expires_at = cached
        if time.time() < expires_at:
            return RSAAlgorithm.from_jwk(pem)

    jwks = _fetch_jwks(jwks_url)
    for key in jwks:
        if key.get("kid") == kid:
            public_key = RSAAlgorithm.from_jwk(key)
            pem = public_key.public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
            ).decode("utf-8")
            _jwks_cache[kid] = (pem, time.time() + _cache_ttl)
            return public_key

    raise InvalidSignatureError(f"Key ID '{kid}' not found in JWKS")


def verify_clerk_token(
    token: str,
    config: ClerkConfig,
) -> dict:
    """Verify a Clerk-issued JWT and return the payload.

    Raises HTTP-exception equivalents (DecodeError, etc.) on failure.
    """
    jwks_url = _get_jwks_url(config.domain)

    try:
        public_key = _get_public_key(token, jwks_url)
    except httpx.HTTPError as exc:
        raise DecodeError(f"Failed to fetch JWKS: {exc}") from exc

    payload = PyJWT().decode(
        token,
        key=public_key,
        algorithms=["RS256"],
        options={
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
            "verify_aud": bool(config.audience),
            "verify_iss": bool(config.domain),
        },
        audience=config.audience if config.audience else None,
        issuer=config.domain if config.domain else None,
    )

    return payload
