"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""

import traceback
from pathlib import Path
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# from fastapi.templating import Jinja2Templates # Moved to templating.py
from fastapi.staticfiles import StaticFiles

# Logging configuration must be initialized early
from prompthelix.config import settings
from prompthelix.logging_config import configure_logging
from prompthelix.templating import templates # Import templates object

from prompthelix.api import routes as api_routes
from prompthelix.ui_routes import router as ui_router  # Import the UI router
from prompthelix import metrics as ph_metrics

# from prompthelix.websocket_manager import ConnectionManager # No longer imported directly for instantiation
from prompthelix.globals import websocket_manager  # Import the global instance
from prompthelix.database import init_db
from prompthelix.logging_config import setup_logging # Import the logging setup function

# --- Setup Logging ---
# Call this early, before other initializations if they might log.
setup_logging()
# --- End Setup Logging ---


from prompthelix.utils import setup_logging

setup_logging()

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging as soon as possible
configure_logging(settings.DEBUG)


# Call init_db to create database tables on startup
# For production, you'd likely use Alembic migrations separately.
# init_db() # This should be commented out for tests; conftest.py handles DB setup.
# For running the app directly (e.g. `python -m prompthelix.main`),
# it might be called below if __name__ == "__main__".

# Initialize FastAPI application
app = FastAPI()
# websocket_manager = ConnectionManager() # websocket_manager is now imported from globals

# Mount static files
# templates object is now imported, no need to initialize here
# Mount static files using an absolute path so the server can be started from
# any working directory.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Add this before including routers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # In a production environment, you would typically check a DEBUG flag
    # (e.g., from settings) to decide whether to include the traceback.
    # For example:
    # from prompthelix.config import settings
    # if settings.DEBUG:
    #     return JSONResponse(
    #         status_code=500,
    #         content={
    #             "message": "An unexpected error occurred.",
    #             "detail": str(exc),
    #             "traceback": traceback.format_exc(),
    #         },
    #     )
    # else:
    #     # Log the exception for server-side review
    #     # logger.error(f"Unhandled exception: {exc}", exc_info=True)
    #     return JSONResponse(
    #         status_code=500,
    #         content={
    #             "message": "An unexpected server error occurred.",
    #             "detail": "Please contact support or try again later.",
    #         },
    #     )

    # For now, always include traceback as per user request for easier debugging in current context.
    # Consider adding a DEBUG flag check for production.
    print(f"Unhandled exception: {exc}")  # Basic logging
    traceback.print_exc()  # Print traceback to server console

    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred.",
            "detail": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


# TODO: Implement WebSocket support for real-time communication (future feature).
# TODO: Implement robust authentication and authorization mechanisms (future feature).
# TODO: Implement rate limiting to protect the API (future feature).


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for dashboard real-time updates.
    """
    await websocket_manager.connect(websocket)
    try:
        from prompthelix import globals as ph_globals
        await websocket_manager.send_personal_json({"type": "ga_history", "data": ph_globals.ga_history}, websocket)
    except Exception as e:  # pragma: no cover - simple log
        print(f"Failed to send GA history: {e}")
    await websocket_manager.broadcast_json({"message": "A new client has connected!"})
    try:
        while True:
            data = await websocket.receive_text()
            print(f"WebSocket dashboard received: {data}")
            await websocket_manager.send_personal_json(
                {"response": f"You wrote: {data}"}, websocket
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        print("WebSocket dashboard disconnected")
        await websocket_manager.broadcast_json(
            {"message": "A client has disconnected."}
        )
    except Exception as e:
        websocket_manager.disconnect(websocket)
        print(f"WebSocket dashboard error: {e}")
        # It's good practice to try and close the websocket if it's still open during an unexpected error.
        # However, manager.disconnect already removed it from the list.
        # Depending on the error, the websocket might already be closed or in an unusable state.
        # For now, we'll rely on the client or server to eventually clean up the connection.
        # Consider await websocket.close(code=1011) if appropriate for specific errors.
        await websocket_manager.broadcast_json(
            {"message": f"A client connection had an error: {type(e).__name__}"}
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """
    Root endpoint for the PromptHelix API.
    Returns a welcome message.
    """
    return {"message": "Welcome to PromptHelix API"}

from prometheus_client import generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from fastapi.responses import Response # Ensure Response is imported

@app.get("/metrics", name="prometheus_metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# Include API routes
app.include_router(api_routes.router)
# Include UI routes
app.include_router(ui_router, prefix="/ui", tags=["UI"])

if __name__ == "__main__":
    # This block is for when you run the application directly, e.g., using `python -m prompthelix.main`
    # It's a good place to initialize the database if it hasn't been set up by other means (like Alembic).
    # init_db() # Uncomment if you want to ensure DB is created/checked when running directly.
    # However, be cautious if you use Alembic for migrations, as this might conflict.
    # For development, manually running `init_db()` via a script or an initial check might be safer.

    # Note: Uvicorn is typically used to run the app, e.g., `uvicorn prompthelix.main:app --reload`
    # In that case, this __main__ block might not be executed depending on how uvicorn imports/runs the app.
    # If `init_db()` is critical on every startup when not testing, ensure it's called appropriately,
    # possibly earlier in the script if not managed by a migration tool or separate startup script.
    pass  # Placeholder if no direct run actions are needed here right now.
