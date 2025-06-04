"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""
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
