from datetime import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session as DbSession

from prompthelix.database import get_db
from prompthelix.models.user_models import User as UserModel
from prompthelix.services import user_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: DbSession = Depends(get_db)) -> UserModel:
    print(f"get_current_user called with token (first 8 chars): {token[:8]}...")
    db_session = user_service.get_session_by_token(db, session_token=token)
    if not db_session:
        print(f"No session found for token (first 8 chars): {token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    print(f"Session found for token (first 8 chars): {token[:8]}..., expires at: {db_session.expires_at}")
    if db_session.expires_at < datetime.utcnow():
        print(f"Session expired for token (first 8 chars): {token[:8]}... at {db_session.expires_at}")
        user_service.delete_session(db, session_token=token) # Attempt to clean up expired session
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_service.get_user(db, user_id=db_session.user_id)
    if user is None:
        print(f"User not found for session's user_id: {db_session.user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for session",
            headers={"WWW-Authenticate": "Bearer"},
        )

    print(f"User {user.username} (ID: {user.id}) successfully authenticated via token (first 8 chars): {token[:8]}...")
    return user
