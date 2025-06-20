"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.templating import Jinja2Templates # Moved to templating.py
from fastapi.staticfiles import StaticFiles
from prompthelix.templating import templates # Import templates object
from prompthelix.api import routes as api_routes
from prompthelix.ui_routes import router as ui_router # Import the UI router
# from prompthelix.websocket_manager import ConnectionManager # No longer imported directly for instantiation
from prompthelix.globals import websocket_manager # Import the global instance
from prompthelix.database import init_db

# Call init_db to create database tables on startup
# For production, you'd likely use Alembic migrations separately.
init_db()

# Initialize FastAPI application
app = FastAPI()
# websocket_manager = ConnectionManager() # websocket_manager is now imported from globals

# Mount static files
# templates object is now imported, no need to initialize here
app.mount("/static", StaticFiles(directory="prompthelix/static"), name="static")


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
    print(f"Unhandled exception: {exc}") # Basic logging
    traceback.print_exc() # Print traceback to server console

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
    await websocket_manager.broadcast_json({"message": "A new client has connected!"})
    try:
        while True:
            data = await websocket.receive_text()
            print(f"WebSocket dashboard received: {data}")
            await websocket_manager.send_personal_json({"response": f"You wrote: {data}"}, websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        print("WebSocket dashboard disconnected")
        await websocket_manager.broadcast_json({"message": "A client has disconnected."})
    except Exception as e:
        websocket_manager.disconnect(websocket)
        print(f"WebSocket dashboard error: {e}")
        # It's good practice to try and close the websocket if it's still open during an unexpected error.
        # However, manager.disconnect already removed it from the list.
        # Depending on the error, the websocket might already be closed or in an unusable state.
        # For now, we'll rely on the client or server to eventually clean up the connection.
        # Consider await websocket.close(code=1011) if appropriate for specific errors.
        await websocket_manager.broadcast_json({"message": f"A client connection had an error: {type(e).__name__}"})


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
app.include_router(ui_router)
