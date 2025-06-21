import os
import importlib
import inspect
from typing import List, Optional
from pathlib import Path
import asyncio

from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query, status
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from starlette.status import HTTP_303_SEE_OTHER  # For POST redirect
import httpx  # For making API calls from UI routes
from datetime import datetime
import unittest
import io

from prompthelix.templating import templates  # Import from templating.py
from prompthelix.database import get_db      # Ensure this is imported
from prompthelix.api import crud             # Ensure this is imported
from prompthelix import schemas              # Import all schemas
from prompthelix.enums import ExecutionMode  # Added import
from prompthelix.agents.base import BaseAgent
from prompthelix.models.user_models import User as UserModel
from prompthelix.services import user_service


router = APIRouter()

SUPPORTED_LLM_SERVICES = [
    {"name": "OPENAI", "display_name": "OpenAI", "description": "API key for OpenAI models (e.g., GPT-4, GPT-3.5)."},
    {"name": "ANTHROPIC", "display_name": "Anthropic", "description": "API key for Anthropic models (e.g., Claude)."},
    {"name": "GOOGLE", "display_name": "Google", "description": "API key for Google AI models (e.g., Gemini)."}
]


async def get_current_user_ui(
    request: Request, db: Session = Depends(get_db)
) -> UserModel:
    token = request.cookies.get("prompthelix_access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    db_session = user_service.get_session_by_token(db, session_token=token)
    if not db_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    if db_session.expires_at < datetime.utcnow():
        user_service.delete_session(db, session_token=token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    user = user_service.get_user(db, user_id=db_session.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for session")
    return user

def list_available_agents() -> List[dict[str, str]]:  # Updated type hint
    agents_info = []
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(current_
