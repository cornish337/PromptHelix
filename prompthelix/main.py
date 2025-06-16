"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
# from fastapi.templating import Jinja2Templates # Moved to templating.py
from fastapi.staticfiles import StaticFiles
from prompthelix.templating import templates # Import templates object
from prompthelix.api import routes as api_routes
from prompthelix.ui_routes import router as ui_router # Import the UI router
from prompthelix.database import init_db

# Call init_db to create database tables on startup
# For production, you'd likely use Alembic migrations separately.
init_db()

# Initialize FastAPI application
app = FastAPI()

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
