# PromptHelix Project Architecture

## 1. Overview

PromptHelix is a Python-based framework designed for the advanced generation, optimization, and management of AI prompts. It leverages a combination of genetic algorithms, a multi-agent system, and integration with various Large Language Models (LLMs) to facilitate sophisticated prompt engineering. The system provides a FastAPI backend for API interactions and a basic web UI, along with a command-line interface (CLI) for core operations.

## 2. Core Components

This section details the major functional blocks of the PromptHelix system.

### 2.1. FastAPI Backend
*   **Purpose**: Provides the main application interface through RESTful APIs and serves a web-based UI.
*   **Key Modules/Files**:
    *   `prompthelix/main.py`: Main FastAPI application setup, middleware, exception handlers, and WebSocket endpoint for dashboard.
    *   `prompthelix/api/routes.py`: Defines all primary API endpoints (prompts, users, experiments, settings, LLM utilities).
    *   `prompthelix/api/crud.py`: Contains Create, Read, Update, Delete operations, acting as a database interaction layer for API routes.
    *   `prompthelix/ui_routes.py`: Defines routes for serving HTML pages.
    *   `prompthelix/templates/`: HTML templates for the web UI.
    *   `prompthelix/static/`: Static assets (CSS, JavaScript) for the UI.

### 2.2. Configuration System
*   **Purpose**: Manages all application settings, including sensitive keys, database connections, and operational parameters.
*   **Key Modules/Files**:
    *   `prompthelix/config.py`: Central configuration file. Loads settings from environment variables and `.env` files. Defines default agent settings, LLM parameters, and the crucial `AGENT_PIPELINE_CONFIG` which determines the active agents and their setup.
    *   `.env.example`: Template for the `.env` file.

### 2.3. Database System
*   **Purpose**: Handles persistent storage of application data.
*   **Key Modules/Files**:
    *   `prompthelix/database.py`: Configures the SQLAlchemy engine, session management (`SessionLocal`), and includes `init_db()` for table creation (primarily for development).
    *   `prompthelix/models/`: Contains all SQLAlchemy ORM models defining the database schema (e.g., `User`, `Prompt`, `PromptVersion`, `LLMStatistic`).
    *   `alembic/`: Directory containing Alembic migration scripts for managing database schema changes in production.
    *   `alembic.ini`: Alembic configuration file.

### 2.4. Agent System
*   **Purpose**: Implements a multi-agent architecture where specialized agents collaborate on prompt engineering tasks.
*   **Key Modules/Files**:
    *   `prompthelix/agents/base.py`: Defines the `BaseAgent` abstract class with common functionalities like message sending/receiving and metric publishing.
    *   `prompthelix/agents/`: Directory containing specific agent implementations (e.g., `PromptArchitectAgent`, `ResultsEvaluatorAgent`, `StyleOptimizerAgent`, `MetaLearnerAgent`). Each agent has a specialized role.
    *   `prompthelix/message_bus.py`: Facilitates communication between agents. Agents can subscribe to message types and broadcast messages.
    *   `knowledge/`: Directory where some agents (e.g., `DomainExpertAgent`, `MetaLearnerAgent`) persist their knowledge or state as JSON files.

### 2.5. Genetic Algorithm (GA) Engine
*   **Purpose**: The core engine for optimizing prompts using evolutionary principles.
*   **Key Modules/Files**:
    *   `prompthelix/orchestrator.py` (`main_ga_loop` function): Orchestrates the setup and execution of GA runs. Initializes agents, GA components, and manages the overall flow.
    *   `prompthelix/experiment_runners/ga_runner.py` (`GeneticAlgorithmRunner` class): Manages the generation-by-generation evolutionary loop, including controls for pausing, resuming, and stopping.
    *   `prompthelix/genetics/engine.py`:
        *   `PromptChromosome`: Represents an individual prompt in the GA population.
        *   `GeneticOperators`: Implements selection, crossover, and mutation operations.
        *   `FitnessEvaluator`: Evaluates the fitness of `PromptChromosome` instances, often by delegating to a `ResultsEvaluatorAgent`.
        *   `PopulationManager`: Manages the collection of `PromptChromosome`s, including initialization, evolution tracking, and persistence.
    *   `prompthelix/genetics/*_strategies.py`: Modules defining different strategies for mutation, crossover, and selection.

### 2.6. Services Layer
*   **Purpose**: Encapsulates business logic and abstracts direct database interactions from the API routes or other parts of the application.
*   **Key Modules/Files**:
    *   `prompthelix/services/`: Directory containing service modules like `prompt_service.py`, `user_service.py`, `performance_service.py`, `evolution_service.py`. These services often use `crud.py` functions for database access.

### 2.7. LLM Utilities
*   **Purpose**: Provides tools and abstractions for interacting with various Large Language Models.
*   **Key Modules/Files**:
    *   `prompthelix/utils/llm_utils.py`: Contains functions for making API calls to different LLM providers (OpenAI, Anthropic, Google), managing API keys, and listing available LLMs.

### 2.8. Real-time Updates & Monitoring
*   **Purpose**: Provides real-time feedback and monitoring capabilities, especially for GA progress.
*   **Key Modules/Files**:
    *   `prompthelix/websocket_manager.py`: Manages WebSocket connections for broadcasting updates (e.g., to the UI dashboard).
    *   `prompthelix/globals.py`: Defines global instances like `websocket_manager`.
    *   `prompthelix/metrics.py` & Prometheus integration: Exposes application and GA metrics via a `/metrics` endpoint for Prometheus scraping.
    *   `prompthelix/wandb_logger.py` (and W&B integration in `orchestrator.py`): Logs experiment metrics to Weights & Biases if configured.

### 2.9. Task Queuing & Background Jobs
*   **Purpose**: Enables asynchronous execution of long-running tasks, such as GA experiments.
*   **Key Modules/Files**:
    *   FastAPI `BackgroundTasks`: Used in `prompthelix/api/routes.py` for running GA experiments asynchronously when triggered via the API.
    *   `prompthelix/celery_app.py`: Contains setup for Celery, suggesting its availability for more distributed task queuing, though its active integration level for core GA runs triggered via API seems to have shifted to `BackgroundTasks`.

### 2.10. Command-Line Interface (CLI)
*   **Purpose**: Offers a command-line tool for interacting with the system, primarily for running GA experiments, tests, and utility functions.
*   **Key Modules/Files**:
    *   `prompthelix/cli.py`: Implements the CLI using a library like Typer or Click. Handles commands like `run ga`, `test`, `check-llm`.

## 3. Key Flows

This section describes the sequence of operations for critical system processes.

### 3.1. Genetic Algorithm Experiment Run

This flow outlines how a new prompt is generated and optimized using the GA.

1.  **Initiation**:
    *   **Via API**: A user sends a POST request to `/api/experiments/run-ga` with parameters like task description, keywords, GA settings (population size, generations), and optionally an initial prompt or parent prompt ID.
    *   **Via CLI**: A user runs `python -m prompthelix.cli run ga` with similar parameters.

2.  **Orchestration Setup (`prompthelix.orchestrator.main_ga_loop`)**:
    *   The `main_ga_loop` function is invoked.
    *   **Message Bus**: Initializes the `MessageBus` for inter-agent communication.
    *   **Agent Loading**: Loads and instantiates agents (e.g., `PromptArchitectAgent`, `ResultsEvaluatorAgent`, `StyleOptimizerAgent`) based on `AGENT_PIPELINE_CONFIG` from `config.py`. Agents are registered with the message bus.
    *   **GA Components Instantiation**:
        *   `GeneticOperators`: For selection, crossover, mutation.
        *   `FitnessEvaluator`: Configured to evaluate prompt fitness (often using the `ResultsEvaluatorAgent`).
        *   `PopulationManager`: To manage the prompt population, configured with GA parameters and the initial prompt if provided.
    *   **W&B Logging**: Initializes Weights & Biases logging if configured.
    *   **Prometheus Metrics**: Ensures Prometheus metrics exporter is running if enabled.

3.  **Population Initialization (`PopulationManager.initialize_population`)**:
    *   If no existing population is loaded or an initial prompt string is not solely defining the population:
        *   The `PromptArchitectAgent` is invoked to generate an initial set of `PromptChromosome`s based on the task description and keywords.
    *   If an `initial_prompt_str` is provided, it's used to create one or more starting chromosomes.
    *   The initial population is evaluated by the `FitnessEvaluator`.

4.  **Evolutionary Loop (`GeneticAlgorithmRunner.run`)**:
    *   The `GeneticAlgorithmRunner` takes control of the generation-by-generation evolution.
    *   For each generation:
        *   **Evaluation**: The `FitnessEvaluator` assesses the fitness of each `PromptChromosome` in the current population. This may involve:
            *   The `ResultsEvaluatorAgent` processing the prompt (potentially by sending it to an LLM and analyzing the output against criteria).
            *   The agent then returns a fitness score.
        *   **Metrics Update**: Generation metrics (best/mean/min fitness) are calculated and logged (DB, W&B, WebSocket broadcast via `PopulationManager.broadcast_ga_update`).
        *   **Selection**: `GeneticOperators.selection` selects parent chromosomes for breeding (e.g., using tournament selection).
        *   **Crossover**: `GeneticOperators.crossover` combines pairs of selected parents to produce offspring.
        *   **Mutation**: `GeneticOperators.mutation` introduces variations into the offspring. This step can involve:
            *   Applying various mutation strategies (e.g., character append, slice reverse).
            *   Optionally, the `StyleOptimizerAgent` might be invoked to refine the mutated prompt's style.
        *   **New Population**: A new population is formed from elite individuals (best from the previous generation) and the newly generated offspring.
        *   **Pause/Stop Check**: The runner checks for pause or stop signals.
        *   **Persistence**: The `PopulationManager` may save the current population state periodically (e.g., every N generations to a JSON file).

5.  **Completion**:
    *   The loop continues for the specified number of generations or until a stop condition is met.
    *   The `GeneticAlgorithmRunner` returns the best `PromptChromosome` found during the run.
    *   `main_ga_loop` may perform final logging (e.g., to W&B) and save the final population state.
    *   If initiated via the API background task, the result (best prompt) is typically saved as a new `PromptVersion` linked to a `Prompt` in the database.

### 3.2. User Interaction & Prompt Management (API)

This flow describes typical user interactions with the system via its API.

1.  **Authentication**:
    *   User requests an access token from `/auth/token` by providing username and password.
    *   The system validates credentials and, if successful, returns a JWT (session token).
    *   Subsequent authenticated requests must include this token in the `Authorization` header.

2.  **Prompt Creation/Management**:
    *   **Create Prompt**: User sends POST to `/api/prompts` with prompt details (name, description). A new `Prompt` record is created, owned by the authenticated user.
    *   **List Prompts**: User sends GET to `/api/prompts` to retrieve a list of prompts.
    *   **Create Prompt Version**: User sends POST to `/api/prompts/{prompt_id}/versions` with content for a new version of an existing prompt. A `PromptVersion` record is created.
    *   **View/Update/Delete**: Users can view specific prompts/versions (GET), update them (PUT), or delete them (DELETE), subject to ownership checks.

3.  **LLM Interaction**:
    *   **Test Prompt**: User sends POST to `/api/llm/test_prompt` with prompt text and desired LLM service. The system calls the LLM and returns the response.
    *   **View LLM Stats**: User sends GET to `/api/llm/statistics` to see usage counts for different LLMs.

### 3.3. Agent Communication Flow

Agents collaborate and exchange information using the `MessageBus`.

1.  **Registration**: On startup (within `main_ga_loop`), agents are instantiated and registered with the `MessageBus`.
2.  **Subscription**: Agents can subscribe to specific message types they are interested in (e.g., `MetaLearnerAgent` subscribes to `evaluation_result` and `critique_result`).
3.  **Message Broadcasting**:
    *   An agent performs an action (e.g., `ResultsEvaluatorAgent` finishes scoring a prompt).
    *   It calls `message_bus.broadcast_message(message_type, payload, sender_id)`.
4.  **Message Dispatch**:
    *   The `MessageBus` identifies all agents subscribed to that `message_type`.
    *   It calls the `receive_message(message)` method on each subscribed agent.
5.  **Message Handling**:
    *   The receiving agent's `receive_message` method processes the payload. This might trigger further actions, internal state updates (e.g., `MetaLearnerAgent` updates its knowledge base), or new messages being broadcast.
    *   Direct messages are also possible using `agent.send_message()`, which routes through the bus to a specific recipient.
6.  **Logging**: The `MessageBus` can log messages to the database if configured with a `db_session_factory`.
7.  **WebSocket Updates**: The `MessageBus` is also connected to the `WebSocketManager` (via `PopulationManager` or directly by agents publishing metrics) to send real-time updates (e.g., GA progress, agent metrics) to connected UI clients.

## 4. Data Persistence

This section covers how and where the application stores data.

### 4.1. Relational Database
*   **Technology**: SQL-based relational database (e.g., PostgreSQL for production, SQLite for development).
*   **ORM**: SQLAlchemy is used as the Object-Relational Mapper, allowing interaction with the database using Python objects.
    *   `prompthelix/database.py`: Configures the database engine (`create_engine`) and session factory (`sessionmaker`).
    *   `prompthelix/models/`: Defines the database schema through SQLAlchemy models. Key models include:
        *   `User`, `Session`: For user authentication and session management.
        *   `Prompt`, `PromptVersion`: For storing prompts and their iterative versions.
        *   `PerformanceMetric`: For tracking performance metrics of prompt versions.
        *   `LLMStatistic`: For logging usage of different LLM services.
        *   `APIKey`: For storing API keys for external services (LLMs) in the database.
        *   `ConversationLog`: For logging inter-agent messages and other system interactions.
        *   `GAExperimentRun`, `GAGenerationMetric`, `GAChromosome`: For tracking GA experiment details, per-generation metrics, and individual chromosomes.
*   **Schema Migrations**: Alembic is used for managing database schema evolution.
    *   `alembic/`: Contains migration scripts.
    *   `alembic.ini`: Alembic configuration.
    *   Developers use `alembic revision` to generate new migration scripts and `alembic upgrade head` to apply them.
*   **Initialization**: For development, `prompthelix.database.init_db()` can create all tables based on models if they don't exist (suitable for SQLite).

### 4.2. Agent Knowledge & State
*   **Format**: JSON files.
*   **Location**: Typically stored within the `knowledge/` directory at the project root, or a path configured via `KNOWLEDGE_DIR`.
*   **Usage**:
    *   Agents like `DomainExpertAgent`, `MetaLearnerAgent`, and `PromptCriticAgent` can persist their internal knowledge bases, rules, or learned parameters to JSON files.
    *   This allows their state to survive application restarts and be version-controlled if desired.
    *   File paths are often configurable via agent settings in `prompthelix/config.py` or environment variables.

### 4.3. Genetic Algorithm Population
*   **Format**: JSON file.
*   **Location**: Configurable via `DEFAULT_POPULATION_PERSISTENCE_PATH` in `prompthelix/config.py` (defaults to `knowledge/ga_population.json`) or overridden at runtime.
*   **Usage**:
    *   The `PopulationManager` can save the current state of the GA population (all `PromptChromosome`s, generation number) to this file.
    *   This allows GA runs to be resumed or their final states to be analyzed.
    *   Saving can occur periodically during a run (controlled by `DEFAULT_SAVE_POPULATION_FREQUENCY`) and/or at the end of the run.

### 4.4. Application Logs
*   **Format**: Plain text or JSON lines.
*   **Location**: Configurable via `LOG_DIR` and `LOG_FILE_NAME` in `prompthelix/config.py`. Defaults to a `logs/` directory.
*   **Usage**: Standard application logging captures events, errors, and debug information. Structured logging (JSON) can be enabled for easier parsing by log management systems.

## 5. Configuration Management

This section explains how the application is configured.

### 5.1. Central Configuration File
*   **File**: `prompthelix/config.py`
*   **Purpose**: Acts as the primary source for application settings. It defines a `Settings` class (or a similar structure) that aggregates configuration variables.
*   **Key Settings Managed**:
    *   Database connection URL (`DATABASE_URL`).
    *   API keys for LLM services (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). These can also be fetched from the database if stored there.
    *   Debug mode (`DEBUG` or `PROMPTHELIX_DEBUG`).
    *   Redis connection details (`REDIS_HOST`, `REDIS_PORT`) for caching or Celery.
    *   Celery broker and backend URLs (`CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`).
    *   Default GA parameters (population size, mutation rate, etc.).
    *   Paths for knowledge persistence (`KNOWLEDGE_DIR`), log files (`LOG_DIR`).
    *   Agent-specific settings (default models, file paths, behavior flags) in `AGENT_SETTINGS`.
    *   The `AGENT_PIPELINE_CONFIG` JSON string, which defines the sequence and configuration of active agents in the GA orchestration.
    *   Paths to GA strategy classes (mutation, selection, crossover).
    *   Experiment tracking settings (W&B API key, project name).

### 5.2. Environment Variables
*   **Priority**: Environment variables typically override values defined directly in `config.py` or loaded from `.env` files (due to `load_dotenv(override=True)`).
*   **Usage**:
    *   The primary way to provide sensitive information (like API keys, database passwords) in production environments.
    *   Used to set operational modes (e.g., `PROMPTHELIX_DEBUG=true`).
    *   Can override specific agent settings using a convention like `<AGENTNAME>_<SETTING>=value`.
    *   The `DATABASE_URL` is almost always set via an environment variable.

### 5.3. `.env` File
*   **Purpose**: Allows developers to set environment variables locally for development without modifying shell configurations.
*   **Loading**: `python-dotenv` is used in `prompthelix/config.py` to load variables from a `.env` file located in the project root.
*   **Best Practice**: `.env` should be included in `.gitignore` to prevent committing sensitive credentials. An `.env.example` file is provided as a template.

### 5.4. Agent Configuration
*   **Global Defaults**: `prompthelix/config.py` contains an `AGENT_SETTINGS` dictionary providing default configurations for various agent types.
*   **Environment Overrides**: These defaults can be overridden by environment variables (e.g., `PROMPTARCHITECT_DEFAULT_LLM_MODEL="gpt-4"`).
*   **Pipeline Configuration (`AGENT_PIPELINE_CONFIG`)**: A JSON string (typically set via an environment variable `AGENT_PIPELINE_CONFIG_JSON`) defines which agents are active in the `main_ga_loop`, their IDs, and which key from `AGENT_SETTINGS` (plus overrides) to use for their configuration. This allows dynamic assembly of the agent pipeline.
*   **Runtime Overrides**: The `main_ga_loop` function in `orchestrator.py` accepts `agent_settings_override` and `llm_settings_override` parameters, allowing specific GA runs to use custom settings for agents or LLM interactions.

### 5.5. Logging Configuration
*   **Module**: `prompthelix/logging_config.py` (and settings in `prompthelix/config.py`).
*   **Settings**: Log level, log file path, log format are configurable via `config.py` and environment variables (e.g., `PROMPTHELIX_LOG_LEVEL`, `PROMPTHELIX_LOG_FILE`).
*   **Setup**: `setup_logging()` is called early in application startup (`main.py`, `cli.py`) to apply these settings.

## 6. Potential Areas for Improvement & Redundancies

This section lists observations about parts of the codebase that might be redundant, obsolete, or could be improved for robustness, scalability, and maintainability.

### 6.1. Code Redundancies & Obsolete Code
1.  **Logging Setup in `prompthelix/main.py`**:
    *   Multiple `setup_logging()` calls from different locations (`prompthelix.logging_config` and `prompthelix.utils.logging_utils` which might be the same). This should be consolidated into a single, clear logging initialization sequence, likely driven by `prompthelix.logging_config.py`.
2.  **Duplicate `/metrics` Endpoint in `prompthelix/main.py`**:
    *   Two `@app.get("/metrics")` route handlers are defined. The second one, using `prometheus_client.generate_latest`, is standard and should be kept; the other should be removed.
3.  **Commented-Out Code**:
    *   The `/debug-routes` endpoint in `prompthelix/main.py` is commented out. If not needed, it should be removed.
    *   Various smaller commented-out lines or blocks exist across files. These should be reviewed and removed if obsolete.
4.  **Duplicate `DEBUG` Setting in `prompthelix/config.py`**:
    *   The `DEBUG` attribute within the `Settings` class in `config.py` is defined twice. One instance should be removed.
5.  **Duplicate Agent Loading Loop in `prompthelix.orchestrator.py`**:
    *   The code block for instantiating agents from `AGENT_PIPELINE_CONFIG` appears to be duplicated. The first instance seems incomplete and should be removed or merged.
6.  **Unused `metrics_logger` in `prompthelix.orchestrator.py`**:
    *   A `metrics_logger_instance` is created but then the `metrics_logger` argument is commented out in the `GeneticOperators` instantiation. `GeneticOperators` class does not expect this argument. This can be removed.

### 6.2. Configuration & Initialization
1.  **`ensure_directories_exist()` Call in `prompthelix/config.py`**:
    *   The function `ensure_directories_exist` is defined but its call is commented out. This function is useful for ensuring `KNOWLEDGE_DIR` and `LOG_DIR` exist at startup and should be called appropriately (e.g., in `main.py` or `cli.py` during app initialization).
2.  **Clarity of Logging Configuration**:
    *   While `prompthelix/config.py` defines `LOGGING_CONFIG` dictionary and individual log settings, the actual application of these (e.g., `dictConfig` vs. `basicConfig` vs. manual handler setup in `logging_config.py`) could be clarified for consistency.

### 6.3. GA Concurrency and State Management
1.  **Global GA Runner (`ph_globals.active_ga_runner`)**:
    *   API routes for controlling GA experiments (`/api/ga/pause`, `/resume`, `/cancel`, `/status`) rely on a single global `active_ga_runner` instance. This will not work correctly if multiple GA experiments are intended to run concurrently and be controlled independently (e.g., if multiple users trigger background GA tasks).
    *   **Suggestion**: Each GA task (e.g., managed by FastAPI `BackgroundTasks` or Celery) should have its own `GeneticAlgorithmRunner` instance. A mechanism to map `task_id` to its corresponding runner/status would be needed for the control API endpoints. This might involve a shared dictionary (with appropriate locking) or storing task status/control information in the database.

### 6.4. Database Session Management in Orchestrator
1.  **`SessionLocal` in `prompthelix.orchestrator.py`**:
    *   `SessionLocal` is set as a global variable within the module and updated when `main_ga_loop` is called. While pragmatic for a long-running function that needs to pass sessions around (e.g., to the message bus), it's less conventional than obtaining sessions via dependency injection or context managers scoped to specific operations.
    *   **Suggestion**: For operations within the loop that require a DB session (like logging metrics), consider obtaining a fresh session from `SessionLocal()` within a `try/finally` block or using a context manager to ensure sessions are properly closed. The `MessageBus` already uses a factory pattern, which is good.

### 6.5. Code Clarity and Maintainability
1.  **`if __name__ == "__main__":` in `prompthelix.orchestrator.py`**:
    *   This block contains extensive demonstration and test code for various components. While useful for development, it makes the file very long.
    *   **Suggestion**: Move this demonstration/test code into separate scripts in a `examples/` or `scripts/` directory, or into integration tests. This would keep `orchestrator.py` focused on its core logic.
2.  **Error Handling in `FitnessEvaluator` (Default)**:
    *   The default `FitnessEvaluator` in `prompthelix.genetics.engine.py` might pass an empty `llm_output` to `ResultsEvaluatorAgent` if not in `TEST` mode. This relies on the agent to handle it gracefully.
    *   **Suggestion**: Clarify if the `FitnessEvaluator`'s role includes invoking the LLM or if it strictly evaluates a provided `llm_output`. If the former, it should handle LLM calls in `REAL` mode.

### 6.6. Task Queuing Strategy
1.  **`BackgroundTasks` vs. Celery**:
    *   The API uses FastAPI's `BackgroundTasks` for GA experiments. `celery_app.py` suggests Celery is also available.
    *   **Suggestion**: Clarify the intended roles. `BackgroundTasks` is simpler for in-process background work. Celery is more robust for distributed, multi-worker setups. If Celery is the long-term goal for scalability, GA tasks might eventually migrate to it. For now, `BackgroundTasks` is a reasonable start.

## 7. Directory Structure Overview

This provides a high-level map to the project's main directories:

*   `alembic/`: Database migration scripts (Alembic).
*   `knowledge/`: Default storage for agent knowledge files and GA population persistence.
*   `logs/`: Default directory for application log files.
*   `prompthelix/`: Main application package.
    *   `agents/`: Contains implementations of different agent types.
    *   `api/`: FastAPI route definitions, CRUD operations, and API-specific dependencies.
    *   `docs/`: Project documentation files (like this one, potentially others).
    *   `evaluation/`: Components related to prompt evaluation (though much logic is in `ResultsEvaluatorAgent` and `FitnessEvaluator`).
    *   `experiment_runners/`: Code for running experiments, notably `GeneticAlgorithmRunner`.
    *   `genetics/`: Core genetic algorithm components (chromosomes, operators, engine, strategies).
    *   `models/`: SQLAlchemy database models.
    *   `services/`: Business logic layer abstracting database operations.
    *   `static/`: Static files (CSS, JS) for the web UI.
    *   `templates/`: HTML templates for the web UI.
    *   `tests/`: Automated tests (unit, integration, interactive).
    *   `utils/`: Shared utility functions (e.g., LLM interaction, config utilities, logging helpers).
    *   `__init__.py`, `main.py`, `cli.py`, `config.py`, `database.py`, `orchestrator.py`, etc.: Core application files.
*   `tests/`: An older/alternative top-level test directory (most tests seem to be under `prompthelix/tests/`). This might need consolidation or clarification.
*   `.github/`: GitHub-specific files, like workflow configurations for CI/CD.
*   Root directory: Contains files like `Dockerfile`, `docker-compose.yaml`, `requirements.txt`, `README.md`, `LICENSE`.
