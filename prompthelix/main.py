"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""

import traceback
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from prompthelix import metrics as ph_metrics
from prompthelix.api import routes as api_routes
from prompthelix.database import init_db
from prompthelix.globals import websocket_manager  # Import the global instance
from prompthelix.logging_config import setup_logging  # Prefer setup_logging over configure_logging
from prompthelix.templating import templates  # Import templates object
from prompthelix.ui_routes import router as ui_router  # Import the UI router

# --- Setup Logging ---
# Call this early, before other initializations if they might log.
setup_logging()
# --- End Setup Logging ---


# --- Prometheus Metrics Exporter ---
from prompthelix.utils.metrics_exporter import start_exporter_if_enabled
start_exporter_if_enabled() # Start Prometheus client HTTP server if enabled
# --- End Prometheus Metrics Exporter ---


from prompthelix.utils import setup_logging # This is a duplicate import, setup_logging() was already called

# setup_logging() # Duplicate call

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging as soon as possible
configure_logging(settings.DEBUG)



# Call init_db to create database tables on startup
init_db()  # Initialize database and tables on startup

# Initialize FastAPI application
app = FastAPI()

# Mount static files using an absolute path
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # In production, consider toggling tracebacks using a DEBUG setting.
    print(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred.",
            "detail": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for dashboard real-time updates.
    """
    await websocket_manager.connect(websocket)
    try:
        from prompthelix import globals as ph_globals
        await websocket_manager.send_personal_json(
            {"type": "ga_history", "data": ph_globals.ga_history}, websocket
        )
    except Exception as e:
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
        await websocket_manager.broadcast_json(
            {"message": f"A client connection had an error: {type(e).__name__}"}
        )


@app.get("/metrics", name="prometheus_metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """
    Root endpoint for the PromptHelix API.
    Returns a welcome message.
    """
    return {"message": "Welcome to PromptHelix API"}


# Include API routes
app.include_router(api_routes.router)
# Include UI routes
app.include_router(ui_router, prefix="/ui", tags=["UI"])


if __name__ == "__main__":
    # For direct execution, e.g. python -m prompthelix.main
    pass  # Uvicorn or other entry points should load the app properly.
