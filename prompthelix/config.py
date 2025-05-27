"""
Configuration management for the PromptHelix application.

This file defines the settings for the application, including API keys,
database URLs, and other operational parameters. It supports loading
configurations from environment variables and potentially .env files.
"""
import os
# from pydantic import BaseSettings # Uncomment if Pydantic is used for settings management

# Consider using a library like python-dotenv to load .env files
# Example:
# from dotenv import load_dotenv
# load_dotenv()

class Settings:
    """
    Application settings class.

    Attributes will be loaded from environment variables.
    Pydantic's BaseSettings can be used for automatic validation and loading.
    """
    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/prompthelix_db")

    # LLM API Keys
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") # For Gemini or other Google models

    # Caching (e.g., Redis)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # Task Queue (e.g., Celery)
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # System Parameters for PromptHelix
    # Example: Default settings for the genetic algorithm
    DEFAULT_POPULATION_SIZE: int = int(os.getenv("DEFAULT_POPULATION_SIZE", "50"))
    DEFAULT_MAX_GENERATIONS: int = int(os.getenv("DEFAULT_MAX_GENERATIONS", "100"))
    DEFAULT_MUTATION_RATE: float = float(os.getenv("DEFAULT_MUTATION_RATE", "0.01"))
    # Add other relevant system parameters here

    # Security settings
    # TODO: Uncomment and set a strong, unique SECRET_KEY for production environments.
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key") # For JWT, session management etc.
    # ALGORITHM: str = "HS256" # For JWT

    # class Config:
    #     env_file = ".env" # For Pydantic to load .env file
    #     env_file_encoding = 'utf-8'

# Instantiate the settings
settings = Settings()

# Example of how to access a setting:
# print(settings.DATABASE_URL)
