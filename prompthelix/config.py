"""
Configuration management for the PromptHelix application.

This file defines the settings for the application, including API keys,
database URLs, and other operational parameters. It supports loading
configurations from environment variables and potentially .env files.
"""
# Load environment variables from a .env file if present
import os
import logging
import json
try:
    from sqlalchemy.orm import Session
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Session = None  # type: ignore
from typing import Optional
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False
# from pydantic import BaseSettings # Uncomment if Pydantic is used for settings management

logger = logging.getLogger(__name__)

# --- Directory for persistent knowledge ---
# Define KNOWLEDGE_DIR early as it's used in Settings defaults
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge") # Relative to project root

# Automatically load variables from a .env file in the project root.
# This allows users to define their API keys and other configuration
# settings in a local .env file without exporting them manually.
load_dotenv(override=True)

# IMPORTANT: LLM API Key Configuration
# The system requires API keys for the Large Language Models it interfaces with.
# These keys should be set as environment variables.
# For example, to use OpenAI models, set the OPENAI_API_KEY environment variable:
#
#   export OPENAI_API_KEY='sk-your_openai_api_key_here'
#
# If the required API keys are not found, the relevant LLM calls will fail.
# The FitnessEvaluator, for instance, will not be able to get actual prompt evaluations
# from OpenAI, and will return error messages or default (low) fitness scores.

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


    # Debug flag controls logging verbosity across the application
    DEBUG: bool = os.getenv("PROMPTHELIX_DEBUG", "false").lower() in {"1", "true", "yes"}

    # Optional experiment tracking integrations
    WANDB_API_KEY: str | None = os.getenv("WANDB_API_KEY")
    MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")

    # Debug flag controlling log verbosity
    #DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}


    # Caching (e.g., Redis)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))


    # System Parameters for PromptHelix
    # Example: Default settings for the genetic algorithm
    DEFAULT_POPULATION_SIZE: int = int(os.getenv("DEFAULT_POPULATION_SIZE", "50"))
    DEFAULT_MAX_GENERATIONS: int = int(os.getenv("DEFAULT_MAX_GENERATIONS", "100"))
    DEFAULT_MUTATION_RATE: float = float(os.getenv("DEFAULT_MUTATION_RATE", "0.01"))
    DEFAULT_SESSION_EXPIRE_MINUTES: int = int(os.getenv("SESSION_EXPIRE_MINUTES", "60"))
    # Add other relevant system parameters here

    # Population persistence settings
    DEFAULT_POPULATION_PERSISTENCE_PATH: str = os.getenv("DEFAULT_POPULATION_PERSISTENCE_PATH", os.path.join(KNOWLEDGE_DIR, "ga_population.json"))
    DEFAULT_SAVE_POPULATION_FREQUENCY: int = int(os.getenv("DEFAULT_SAVE_POPULATION_FREQUENCY", "10"))

    # Prometheus metrics
    PROMETHEUS_METRICS_ENABLED: bool = os.getenv("PROMETHEUS_METRICS_ENABLED", "false").lower() == "true"
    PROMETHEUS_METRICS_PORT: int = int(os.getenv("PROMETHEUS_METRICS_PORT", "8001"))

    # Security settings
    # TODO: Uncomment and set a strong, unique SECRET_KEY for production environments.
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key") # For JWT, session management etc.
    # ALGORITHM: str = "HS256" # For JWT

    FITNESS_EVALUATOR_CLASS: str = os.getenv("FITNESS_EVALUATOR_CLASS", "prompthelix.genetics.engine.FitnessEvaluator")

    # Genetic Operator Strategy Configurations
    MUTATION_STRATEGY_CLASSES: str = os.getenv(
        "MUTATION_STRATEGY_CLASSES",
        "prompthelix.genetics.mutation_strategies.AppendCharStrategy,"
        "prompthelix.genetics.mutation_strategies.ReverseSliceStrategy,"
        "prompthelix.genetics.mutation_strategies.PlaceholderReplaceStrategy"
    ) # Comma-separated string
    SELECTION_STRATEGY_CLASS: str = os.getenv("SELECTION_STRATEGY_CLASS", "prompthelix.genetics.selection_strategies.TournamentSelectionStrategy")
    CROSSOVER_STRATEGY_CLASS: str = os.getenv("CROSSOVER_STRATEGY_CLASS", "prompthelix.genetics.crossover_strategies.SinglePointCrossoverStrategy")


    # class Config:
    #     env_file = ".env" # For Pydantic to load .env file
    #     env_file_encoding = 'utf-8'

    # Agent Pipeline Configuration (example, could be loaded from JSON/YAML string in env var)
    # For simplicity, defining a default Python list structure here.
    # In a real setup, this might be a JSON string in an env var parsed at runtime.
    AGENT_PIPELINE_CONFIG_JSON: str = os.getenv(
        "AGENT_PIPELINE_CONFIG_JSON",
        json.dumps([
            {"class_path": "prompthelix.agents.architect.PromptArchitectAgent", "id": "PromptArchitectAgent", "settings_key": "PromptArchitectAgent"},
            {"class_path": "prompthelix.agents.results_evaluator.ResultsEvaluatorAgent", "id": "ResultsEvaluatorAgent", "settings_key": "ResultsEvaluatorAgent"},
            {"class_path": "prompthelix.agents.style_optimizer.StyleOptimizerAgent", "id": "StyleOptimizerAgent", "settings_key": "StyleOptimizerAgent"}
            # Add other agents like Critic, DomainExpert here if they are part of the default pipeline
        ])
    )


# Instantiate the settings
settings = Settings()

# Parse AGENT_PIPELINE_CONFIG_JSON
try:
    AGENT_PIPELINE_CONFIG = json.loads(settings.AGENT_PIPELINE_CONFIG_JSON)
except json.JSONDecodeError:
    logger.error("Failed to parse AGENT_PIPELINE_CONFIG_JSON. Using empty list.")
    AGENT_PIPELINE_CONFIG = []


_openai_key = settings.OPENAI_API_KEY
if _openai_key:
    display_key = f"{_openai_key[:5]}...{_openai_key[-4:] if len(_openai_key) > 9 else ''}"
else:
    display_key = "NOT_SET"
logger.info(f"Loaded OPENAI_API_KEY: {display_key}")
logger.info(f"Default population persistence path: {settings.DEFAULT_POPULATION_PERSISTENCE_PATH}")
logger.info(f"Default save population frequency: {settings.DEFAULT_SAVE_POPULATION_FREQUENCY}")
logger.info(f"Debug logging enabled: {settings.DEBUG}")

# Example of how to access a setting:
# print(settings.DATABASE_URL)

def _get_key_from_db(db_session: Session, service_name: str) -> Optional[str]:
    if not db_session:  # Should not happen if called by the new functions correctly
        return None
    from prompthelix.api import crud  # Imported here to avoid circular import during module load
    key_obj = crud.get_api_key(db_session, service_name=service_name)
    return key_obj.api_key if key_obj else None

def get_openai_api_key(db: Optional[Session] = None) -> Optional[str]:
    if db:
        key_from_db = _get_key_from_db(db, "OPENAI")
        if key_from_db:
            return key_from_db
    return settings.OPENAI_API_KEY

def get_anthropic_api_key(db: Optional[Session] = None) -> Optional[str]:
    if db:
        key_from_db = _get_key_from_db(db, "ANTHROPIC")
        if key_from_db:
            return key_from_db
    return settings.ANTHROPIC_API_KEY

def get_google_api_key(db: Optional[Session] = None) -> Optional[str]:
    if db:
        key_from_db = _get_key_from_db(db, "GOOGLE")
        if key_from_db:
            return key_from_db
    return settings.GOOGLE_API_KEY

# Example of a test key function to demonstrate DB lookup part
def get_test_db_service_key(db: Optional[Session] = None) -> Optional[str]:
    if db:
        key_from_db = _get_key_from_db(db, "TEST_DB_SERVICE")
        if key_from_db:
            return key_from_db
    return os.getenv("TEST_DB_SERVICE_API_KEY") # Fallback to env var

# --- Agent Specific Settings ---
# This dictionary holds configurations for individual agents.
AGENT_SETTINGS = {
    "MetaLearnerAgent": {
        "knowledge_file_path": "meta_learner_knowledge.json", # Filename, will be combined with KNOWLEDGE_DIR
        "default_llm_provider": "openai",
        "persist_knowledge_on_update": True,
        "default_llm_model": "gpt-3.5-turbo",
    },
    "PromptArchitectAgent": {
        "default_llm_provider": "openai",
        "default_llm_model": "gpt-3.5-turbo",
    },
    "ResultsEvaluatorAgent": {
        "default_llm_provider": "openai",
        "evaluation_llm_model": "gpt-4", # Using a more capable model for evaluation
        "fitness_score_weights": {
            "constraint_adherence": 0.6,
            "llm_quality_assessment": 0.4
        },
    },
    "StyleOptimizerAgent": {
        "default_llm_provider": "openai",
        "default_llm_model": "gpt-3.5-turbo",
    },
    "DomainExpertAgent": {
        "default_llm_provider": "openai",
        "default_llm_model": "gpt-3.5-turbo",
    },
    "PromptCriticAgent": {
        "default_llm_provider": "openai",
        "default_llm_model": "gpt-3.5-turbo",
    },
}

# Apply environment variable overrides for agent settings
def _apply_agent_env_overrides(agent_settings: dict) -> dict:
    """Override agent settings using environment variables."""
    for agent_name, settings_dict in agent_settings.items():
        prefix = agent_name.replace("Agent", "").upper()
        for key, default_val in settings_dict.items():
            env_var = f"{prefix}_{key.upper()}"
            if env_var in os.environ:
                raw_val = os.environ[env_var]
                new_val = raw_val
                try:
                    if isinstance(default_val, bool):
                        new_val = raw_val.lower() in {"1", "true", "yes"}
                    elif isinstance(default_val, int) and raw_val.isdigit():
                        new_val = int(raw_val)
                    elif isinstance(default_val, float):
                        new_val = float(raw_val)
                    elif isinstance(default_val, (dict, list)):
                        new_val = json.loads(raw_val)
                except Exception:
                    new_val = raw_val
                settings_dict[key] = new_val
    return agent_settings

AGENT_SETTINGS = _apply_agent_env_overrides(AGENT_SETTINGS)

# --- LLM Utility Settings ---
LLM_UTILS_SETTINGS = {
    "default_timeout": 60,
    "default_retries": 2,
}

# --- Logging Configuration ---
# KNOWLEDGE_DIR is defined at the top, similar logic for LOG_DIR
LOG_DIR = os.getenv("PROMPTHELIX_LOG_DIR", os.path.join(os.getcwd(), "logs")) # Default to a 'logs' directory in CWD
LOG_LEVEL = os.getenv("PROMPTHELIX_LOG_LEVEL", "INFO").upper()
LOG_FILE_NAME = os.getenv("PROMPTHELIX_LOG_FILE", "prompthelix.log") # Set to None or empty string to disable file logging
LOG_FORMAT = os.getenv(
    "PROMPTHELIX_LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
)

# The LOGGING_CONFIG dict can be used by logging.config.dictConfig if more complex setup is needed.
# For now, these individual settings will be used to set up basicConfig and file handlers.
# We'll keep it here for potential future use or if a module wants to reference it.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Important to not disable loggers from libraries
    "formatters": {
        "standard": {
            "format": LOG_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
        # File handler will be added programmatically in logging_config.py if LOG_FILE_NAME is set
    },
    "root": { # Configuring the root logger
        "handlers": ["console"], # Start with console, file handler added if configured
        "level": LOG_LEVEL,
    },
    "loggers": { # Example: Quieten noisy libraries if needed
        "httpx": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "openai": { # Covers openai._base_client etc.
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False
        }
    }
}
""" Old
    "level": "DEBUG" if settings.DEBUG else "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
"""

# --- Experiment Tracking Configuration ---
ENABLE_WANDB_LOGGING = os.getenv("PROMPTHELIX_ENABLE_WANDB", "false").lower() in ("true", "1", "t")
WANDB_PROJECT_NAME = os.getenv("PROMPTHELIX_WANDB_PROJECT", "PromptHelix-Experiments")
WANDB_ENTITY_NAME = os.getenv("PROMPTHELIX_WANDB_ENTITY") # Optional, W&B will use default if not set

# ensure_directories_exist() function was moved up or is handled differently if KNOWLEDGE_DIR is defined early.
# We still need to ensure KNOWLEDGE_DIR and LOG_DIR exist, this can be done at app startup.
# For now, the definition of KNOWLEDGE_DIR is at the top of the file.

def ensure_directories_exist():
    """
    Ensures that directories specified in the config (like KNOWLEDGE_DIR and LOG_DIR) exist.
    """
    dirs_to_check = {
        "KNOWLEDGE_DIR": KNOWLEDGE_DIR,
        "LOG_DIR": LOG_DIR
    }

    for dir_name, dir_path in dirs_to_check.items():
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                # Use logger here once logging is configured, print for now if early execution
                print(f"Config: Created directory for {dir_name}: {os.path.abspath(dir_path)}")
            except OSError as e:
                print(f"Config: Error creating directory {os.path.abspath(dir_path)} for {dir_name}: {e}")
        else:
            if dir_path:
                # Use logger here once logging is configured
                print(f"Config: Directory for {dir_name} ({os.path.abspath(dir_path)}) already exists or was not set.")

# It's good practice to call this early, e.g., when the orchestrator starts,
# or even here, though calling it here means it runs every time config.py is imported.
# For controlled execution, call it from an application entry point.
# ensure_directories_exist()

print("PromptHelix Config: AGENT_SETTINGS and other global settings loaded.")
# To verify API keys are seen by the Settings class (optional, remove for production):
# print(f"Config - OpenAI Key Loaded via Settings: {'Yes' if settings.OPENAI_API_KEY else 'No (Check Environment Variable OPENAI_API_KEY)'}")
# print(f"Config - Anthropic Key Loaded via Settings: {'Yes' if settings.ANTHROPIC_API_KEY else 'No (Check Environment Variable ANTHROPIC_API_KEY)'}")
