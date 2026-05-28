"""FastAPI dependencies for Clerk-authenticated requests."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from rag_system.auth.clerk_auth import verify_clerk_token
from rag_system.config import ClerkConfig

_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_scheme)] = None,
    config: ClerkConfig = ClerkConfig(),
) -> dict:
    """Validate Clerk JWT from Authorization header and return the payload.

    Attaches user info to request.state.user for downstream access.
    Raises 401 on missing or invalid tokens.
    """
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_clerk_token(creds.credentials, config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    request.state.user = payload
    return payload


def get_user_id(user: Annotated[dict, Depends(get_current_user)]) -> str:
    """Extract the Clerk user ID ('sub' claim) from the verified payload."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' claim",
        )
    return user_id
