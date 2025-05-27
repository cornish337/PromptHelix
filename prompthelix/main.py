"""
Main application file for the PromptHelix API.
Initializes the FastAPI application and includes the root endpoint.
"""
from fastapi import FastAPI

# Initialize FastAPI application
app = FastAPI()

# Placeholder for WebSocket support (to be added)
# Placeholder for Authentication (to be added)
# Placeholder for Rate Limiting (to be added)

@app.get("/")
async def root():
    """
    Root endpoint for the PromptHelix API.
    Returns a welcome message.
    """
    return {"message": "Welcome to PromptHelix API"}

# Placeholder for including API routes from prompthelix.api.routes
# For example: from prompthelix.api import routes as api_routes
# app.include_router(api_routes.router)
