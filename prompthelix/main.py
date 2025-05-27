"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""
from fastapi import FastAPI

# Initialize FastAPI application
app = FastAPI()

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

# Include API routes from prompthelix.api.routes
from prompthelix.api import routes as api_routes
app.include_router(api_routes.router, prefix="/api")
