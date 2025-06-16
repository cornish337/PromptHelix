from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DbSession
from prompthelix.database import get_db
from prompthelix.schemas import ConversationLogEntry, ConversationSession
from prompthelix.services.conversation_service import conversation_service
from prompthelix.models.user_models import User as UserModel
from .dependencies import get_current_user

router = APIRouter()

@router.get("/conversations/sessions/", response_model=List[ConversationSession], tags=["Conversations"])
async def list_conversation_sessions(
    db: DbSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
):
    """
    Get a list of all recorded conversation sessions.
    Each session includes a session_id, message count, and timestamps of first/last messages.
    """
    sessions = conversation_service.get_conversation_sessions(db, skip=skip, limit=limit)
    return sessions

@router.get("/conversations/sessions/{session_id}/messages/", response_model=List[ConversationLogEntry], tags=["Conversations"])
async def get_session_messages(
    session_id: str,
    db: DbSession = Depends(get_db),
    skip: int = 0,
    limit: int = 1000,
    current_user: UserModel = Depends(get_current_user),
):
    """
    Get all messages for a specific conversation session_id.
    Messages are ordered by timestamp.
    """
    messages = conversation_service.get_messages_by_session_id(
        db, session_id=session_id, skip=skip, limit=limit
    )
    # Check if the session itself exists by trying to get at least one message.
    # This is slightly different from the original instruction but more direct for checking a specific session's existence.
    if not messages:
        # To confirm the session_id is truly unknown vs. just empty,
        # we can check if there's any log entry at all for this session_id.
        # The service method `get_messages_by_session_id` already returns an empty list if no messages,
        # so we need a way to distinguish "no session found" from "session exists but is empty" (though an empty session is unlikely with current logging).
        # A more robust check might involve querying for the session_id in ConversationLog directly or enhancing the service.
        # For now, if `get_messages_by_session_id` returns empty, we check if the session_id appears in the list of all sessions.

        # Efficient check: try to get just one message for this session ID. If it doesn't exist, then the session ID is not found.
        # This is implicitly handled if `messages` is empty after the call above.
        # To be absolutely sure the session ID itself is the problem, we might need a dedicated service method like `does_session_exist(session_id)`.
        # The current implementation will return an empty list for a non-existent session_id, which is acceptable.
        # To provide a 404, we need to be sure. Let's refine the check:

        # Query for the existence of the session_id by attempting to retrieve just one message (the first one)
        # This is what `first_message_of_session` was intended for.
        # The `messages` list being empty is the primary condition.
        # The check `if not messages and not conversation_service.get_conversation_sessions(db, limit=1)` was to see if *any* sessions exist at all.
        # That's not quite right for a specific session_id 404.

        # Corrected 404 logic:
        # If messages list is empty, it could be an empty session or a non-existent one.
        # To distinguish, we query for the session_id in the aggregation (expensive) or just trust that an empty message list for a session means it's effectively not found for this context.
        # The most straightforward way: if get_messages_by_session_id returns empty, it's either non-existent or has no messages.
        # For the purpose of this API (returning messages), an empty list is a valid response.
        # However, the requirement is to raise 404 if session_id is not found.
        # So, we need to confirm that this session_id has *never* had any messages.

        # Let's try to fetch session details. If the session_id doesn't appear in aggregated sessions, it's a 404.
        # This is still a bit indirect. The simplest is to check if the `messages` list is empty
        # AND if a query for this specific session_id in the ConversationLog table yields nothing.
        # The current `get_messages_by_session_id` already does this. So, if `messages` is empty, we assume 404.

        # Re-evaluating the 404 condition from the prompt:
        # "if not messages and not conversation_service.get_conversation_sessions(db, limit=1):" -- this checks if *any* session exists at all.
        # "first_message_of_session = conversation_service.get_messages_by_session_id(db, session_id=session_id, limit=1)"
        # "if not first_message_of_session: raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")"
        # This logic is sound: if after trying to get messages (even with a limit of 1), none are found for that session_id, then the session_id itself is considered not found.
        # Since `messages` is already the result of `get_messages_by_session_id`, we can just check `if not messages`.

        # The original prompt's 404 logic was:
        # if not messages and not conversation_service.get_conversation_sessions(db, limit=1): # This outer check is too broad
        #    first_message_of_session = conversation_service.get_messages_by_session_id(db, session_id=session_id, limit=1)
        #    if not first_message_of_session: # This is the key check for the specific session_id
        #        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")
        # This can be simplified: if `messages` (which is `get_messages_by_session_id(...)`) is empty, it means no messages were found for this session_id.
        # This is sufficient to declare the session as "not found" for the purpose of returning its messages.
        if not messages:
             # To be absolutely certain this session_id never existed, we could add a specific check.
             # However, for an endpoint that *retrieves messages for a session*, if there are no messages,
             # returning 404 is reasonable if we define "session not found" as "no messages associated with this session_id".
            raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found or session has no messages.")
    return messages

@router.get("/conversations/all_logs/", response_model=List[ConversationLogEntry], tags=["Conversations"])
async def get_all_conversation_logs(
    db: DbSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
):
    """
    Get all conversation logs across all sessions.
    Useful for a raw view or debugging.
    """
    logs = conversation_service.get_all_logs(db, skip=skip, limit=limit)
    return logs
