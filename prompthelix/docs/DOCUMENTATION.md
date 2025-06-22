# PromptHelix Documentation

This document provides an overview of the PromptHelix application, including its API, web interface, and backend services.

## API Documentation

This section details the API endpoints available in PromptHelix.

### Authentication

#### **POST /auth/token**
*   **Summary:** User login.
*   **Description:** Authenticates a user and returns an access token.
*   **Request Body:** `OAuth2PasswordRequestForm` (username, password).
*   **Response:** `schemas.Token` (access_token, token_type).
*   **Backend Services/CRUD functions:**
    *   `user_service.get_user_by_username`
    *   `user_service.verify_password`
    *   `user_service.create_session`

#### **POST /auth/logout**
*   **Summary:** User logout.
*   **Description:** Logs out the current user by invalidating their session token. Requires authentication.
*   **Request Body:** None (token from header).
*   **Response:** Success message.
*   **Backend Services/CRUD functions:**
    *   `user_service.delete_session`

### Users

#### **POST /users/**
*   **Summary:** Create a new user.
*   **Description:** Registers a new user in the system.
*   **Request Body:** `schemas.UserCreate` (username, email, password).
*   **Response:** `schemas.User` (id, username, email, created_at).
*   **Backend Services/CRUD functions:**
    *   `user_service.get_user_by_username`
    *   `user_service.get_user_by_email`
    *   `user_service.create_user`

#### **GET /users/me**
*   **Summary:** Get current user.
*   **Description:** Retrieves the details of the currently authenticated user. Requires authentication.
*   **Request Body:** None.
*   **Response:** `schemas.User` (id, username, email, created_at).
*   **Backend Services/CRUD functions:**
    *   Relies on `dependencies.get_current_user`.

### Prompts

#### **POST /api/prompts**
*   **Summary:** Create a new prompt.
*   **Description:** Creates a new prompt for the authenticated user. Requires authentication.
*   **Request Body:** `schemas.PromptCreate` (name, description).
*   **Response:** `schemas.Prompt` (id, owner_id, created_at, name, description, versions).
*   **Backend Services/CRUD functions:**
    *   `prompt_service.create_prompt`

#### **GET /api/prompts**
*   **Summary:** List all prompts.
*   **Description:** Retrieves a list of all prompts, with optional pagination (skip, limit).
*   **Request Body:** None. Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.Prompt`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompts`

#### **GET /api/prompts/{prompt_id}**
*   **Summary:** Get a specific prompt.
*   **Description:** Retrieves a single prompt by its ID.
*   **Request Body:** None. Path parameter: `prompt_id` (int).
*   **Response:** `schemas.Prompt`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt`

#### **PUT /api/prompts/{prompt_id}**
*   **Summary:** Update a prompt.
*   **Description:** Updates an existing prompt by its ID. User must be the owner. Requires authentication.
*   **Request Body:** `schemas.PromptUpdate` (name, description - all optional). Path parameter: `prompt_id` (int).
*   **Response:** `schemas.Prompt`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt` (for validation)
    *   `prompt_service.update_prompt`

#### **DELETE /api/prompts/{prompt_id}**
*   **Summary:** Delete a prompt.
*   **Description:** Deletes a prompt by its ID. User must be the owner. Requires authentication.
*   **Request Body:** None. Path parameter: `prompt_id` (int).
*   **Response:** `schemas.Prompt` (the deleted prompt).
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt` (for validation)
    *   `prompt_service.delete_prompt`

### Prompt Versions

#### **POST /api/prompts/{prompt_id}/versions**
*   **Summary:** Create a new prompt version.
*   **Description:** Creates a new version for a specific prompt. User must be owner of the parent prompt or have appropriate permissions. Requires authentication.
*   **Request Body:** `schemas.PromptVersionCreate` (content, parameters_used, fitness_score). Path parameter: `prompt_id` (int).
*   **Response:** `schemas.PromptVersion` (id, prompt_id, version_number, created_at, content, parameters_used, fitness_score).
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt` (for validation)
    *   `prompt_service.create_prompt_version`

#### **GET /api/prompt_versions/{version_id}**
*   **Summary:** Get a specific prompt version.
*   **Description:** Retrieves a single prompt version by its ID.
*   **Request Body:** None. Path parameter: `version_id` (int).
*   **Response:** `schemas.PromptVersion`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt_version`

#### **GET /api/prompts/{prompt_id}/versions**
*   **Summary:** List versions for a prompt.
*   **Description:** Retrieves all versions associated with a specific prompt ID, with optional pagination (skip, limit).
*   **Request Body:** None. Path parameter: `prompt_id` (int). Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.PromptVersion`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt_versions_for_prompt`

#### **PUT /api/prompt_versions/{version_id}**
*   **Summary:** Update a prompt version.
*   **Description:** Updates an existing prompt version by its ID. User must have appropriate permissions. Requires authentication.
*   **Request Body:** `schemas.PromptVersionUpdate` (content, parameters_used, fitness_score - all optional). Path parameter: `version_id` (int).
*   **Response:** `schemas.PromptVersion`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.update_prompt_version`

#### **DELETE /api/prompt_versions/{version_id}**
*   **Summary:** Delete a prompt version.
*   **Description:** Deletes a prompt version by its ID. User must have appropriate permissions. Requires authentication.
*   **Request Body:** None. Path parameter: `version_id` (int).
*   **Response:** `schemas.PromptVersion` (the deleted version).
*   **Backend Services/CRUD functions:**
    *   `prompt_service.delete_prompt_version`

### Performance Metrics

#### **POST /api/performance_metrics/**
*   **Summary:** Record a performance metric.
*   **Description:** Records a new performance metric for a specific prompt version. Requires authentication.
*   **Request Body:** `schemas.PerformanceMetricCreate` (prompt_version_id, metric_name, metric_value).
*   **Response:** `schemas.PerformanceMetric` (id, prompt_version_id, created_at, metric_name, metric_value).
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt_version` (for validation)
    *   `performance_service.record_performance_metric`

#### **GET /api/prompt_versions/{prompt_version_id}/performance_metrics/**
*   **Summary:** Get performance metrics for a version.
*   **Description:** Retrieves all performance metrics recorded for a specific prompt version.
*   **Request Body:** None. Path parameter: `prompt_version_id` (int).
*   **Response:** List of `schemas.PerformanceMetric`.
*   **Backend Services/CRUD functions:**
    *   `prompt_service.get_prompt_version` (for validation)
    *   `performance_service.get_metrics_for_prompt_version`

### Experiments (Genetic Algorithm)

#### **POST /api/experiments/run-ga**
*   **Summary:** Run a Genetic Algorithm experiment in the background.
*   **Description:** Starts a Genetic Algorithm to generate and optimize a prompt. The process runs in the background. The best resulting prompt will be saved as a new version under a specified or new prompt. Requires authentication.
*   **Request Body:** `schemas.GAExperimentParams` (task_description, execution_mode, keywords, num_generations, population_size, elitism_count, optional: parent_prompt_id, prompt_name, prompt_description, initial_prompt_str, agent_settings_override, llm_settings_override, parallel_workers, population_path, save_frequency).
*   **Response:** `schemas.GARunResponse` (message, task_id, status_endpoint).
*   **Backend Services/CRUD functions:**
    *   Initiates `run_ga_background_task` which uses:
        *   `main_ga_loop` (orchestrator)
        *   `prompt_service.get_prompt`
        *   `prompt_service.create_prompt`
        *   `prompt_service.create_prompt_version`

#### **POST /api/ga/pause**
*   **Summary:** Pause the running GA experiment.
*   **Description:** Sends a pause request to the currently active GA runner.
*   **Request Body:** None.
*   **Response:** Message indicating request sent or error.
*   **Backend Services/CRUD functions:**
    *   Accesses `ph_globals.active_ga_runner.pause()`
    *   Accesses `ph_globals.active_ga_runner.get_status()`

#### **POST /api/ga/resume**
*   **Summary:** Resume a paused GA experiment.
*   **Description:** Sends a resume request to the currently active GA runner.
*   **Request Body:** None.
*   **Response:** Message indicating request sent or error.
*   **Backend Services/CRUD functions:**
    *   Accesses `ph_globals.active_ga_runner.resume()`
    *   Accesses `ph_globals.active_ga_runner.get_status()`

#### **POST /api/ga/cancel**
*   **Summary:** Cancel the running GA experiment.
*   **Description:** Sends a stop/cancel request to the currently active GA runner.
*   **Request Body:** None.
*   **Response:** Message indicating request sent or error.
*   **Backend Services/CRUD functions:**
    *   Accesses `ph_globals.active_ga_runner.stop()`

#### **GET /api/ga/status**
*   **Summary:** Get the status of the current GA experiment.
*   **Description:** Retrieves the status from the currently active GA runner or returns an IDLE status.
*   **Request Body:** None.
*   **Response:** `schemas.GAStatusResponse` (status, generation, population_size, best_fitness, etc.).
*   **Backend Services/CRUD functions:**
    *   Accesses `ph_globals.active_ga_runner.get_status()`

#### **GET /api/ga/history**
*   **Summary:** Get GA fitness history for a specific run.
*   **Description:** Retrieves generation metrics for a completed GA run.
*   **Request Body:** None. Query parameters: `run_id` (int), `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.GAGenerationMetric`.
*   **Backend Services/CRUD functions:**
    *   `evolution_service.get_generation_metrics_for_run` (via `get_generation_metrics_for_run` import)

#### **GET /api/experiments/runs**
*   **Summary:** List GA experiment runs.
*   **Description:** Return a paginated list of recorded GA experiment runs.
*   **Request Body:** None. Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.GAExperimentRun`.
*   **Backend Services/CRUD functions:**
    *   `evolution_service.get_experiment_runs` (via `get_experiment_runs` import)

#### **GET /api/experiments/runs/{run_id}/chromosomes**
*   **Summary:** Get chromosomes for a GA run.
*   **Description:** Return chromosomes for the specified run, with optional pagination.
*   **Request Body:** None. Path parameter: `run_id` (int). Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.GAChromosome`.
*   **Backend Services/CRUD functions:**
    *   `evolution_service.get_experiment_run` (via `get_experiment_run` import)
    *   `evolution_service.get_chromosomes_for_run` (via `get_chromosomes_for_run` import)

### LLM Utilities

#### **POST /api/llm/test_prompt**
*   **Summary:** Test a prompt with an LLM.
*   **Description:** Sends a given prompt text to a specified LLM service and returns the response. Increments usage statistics for the LLM service.
*   **Request Body:** `schemas.LLMTestRequest` (llm_service, prompt_text).
*   **Response:** `schemas.LLMTestResponse` (llm_service, response_text).
*   **Backend Services/CRUD functions:**
    *   `llm_utils.call_llm_api`
    *   `crud.increment_llm_statistic`

#### **GET /api/llm/statistics**
*   **Summary:** Get LLM usage statistics.
*   **Description:** Retrieves statistics on the usage of different LLM services, such as call counts.
*   **Request Body:** None.
*   **Response:** List of `schemas.LLMStatistic`.
*   **Backend Services/CRUD functions:**
    *   `crud.get_all_llm_statistics`

#### **GET /api/llm/available**
*   **Summary:** List available LLM services.
*   **Description:** Returns a list of LLM service names that are configured and available for use in the system.
*   **Request Body:** None.
*   **Response:** List of strings (LLM service names).
*   **Backend Services/CRUD functions:**
    *   `llm_utils.list_available_llms`

### Settings

#### **POST /api/settings/apikeys/**
*   **Summary:** Upsert (create or update) an API key for a service.
*   **Description:** Saves or updates an API key for a specified LLM service. Requires authentication.
*   **Request Body:** `schemas.APIKeyCreate` (service_name, api_key).
*   **Response:** `schemas.APIKeyDisplay` (id, service_name, api_key_hint, is_set).
*   **Backend Services/CRUD functions:**
    *   `crud.create_or_update_api_key`

#### **GET /api/settings/apikeys/{service_name}**
*   **Summary:** Get API key information for a service.
*   **Description:** Retrieves non-sensitive information about an API key for a service (e.g., if it's set and a hint). Requires authentication.
*   **Request Body:** None. Path parameter: `service_name` (str).
*   **Response:** `schemas.APIKeyDisplay`.
*   **Backend Services/CRUD functions:**
    *   `crud.get_api_key`

### Interactive Tests

#### **POST /api/interactive_tests/run**
*   **Summary:** Run an interactive test.
*   **Description:** Executes a specified interactive pytest file and returns its output.
*   **Request Body:** Form data with `test_name` (str).
*   **Response:** JSON object with `output` (str) and `returncode` (int).
*   **Backend Services/CRUD functions:**
    *   Uses `subprocess.run` to execute pytest.

### Conversations (prefixed with /api/v1)

#### **GET /api/v1/conversations/sessions/**
*   **Summary:** List conversation sessions.
*   **Description:** Get a list of all recorded conversation sessions. Each session includes a session_id, message count, and timestamps of first/last messages. Requires authentication.
*   **Request Body:** None. Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.ConversationSession`.
*   **Backend Services/CRUD functions:**
    *   `conversation_service.get_conversation_sessions`

#### **GET /api/v1/conversations/sessions/{session_id}/messages/**
*   **Summary:** Get messages for a session.
*   **Description:** Get all messages for a specific conversation session_id. Messages are ordered by timestamp. Requires authentication.
*   **Request Body:** None. Path parameter: `session_id` (str). Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.ConversationLogEntry`.
*   **Backend Services/CRUD functions:**
    *   `conversation_service.get_messages_by_session_id`

#### **GET /api/v1/conversations/all_logs/**
*   **Summary:** Get all conversation logs.
*   **Description:** Get all conversation logs across all sessions. Useful for a raw view or debugging. Requires authentication.
*   **Request Body:** None. Query parameters: `skip` (int, optional), `limit` (int, optional).
*   **Response:** List of `schemas.ConversationLogEntry`.
*   **Backend Services/CRUD functions:**
    *   `conversation_service.get_all_logs`

## Web Interface Documentation

This section describes the user-facing web pages in PromptHelix. All routes require authentication unless otherwise specified.

#### **Page URL (Route Name):** `/ (ui_index)` or `/index (ui_index_alt)`
*   **HTTP Methods:** GET
*   **Template:** `index.html`
*   **Purpose:** Serves the main landing page or home page of the application.
*   **Backend/API Interactions:** None directly specified, likely static or redirects.

#### **Page URL (Route Name):** `/login (ui_login)`
*   **HTTP Methods:** GET, POST
*   **Template:** `login.html`
*   **Purpose:**
    *   GET: Displays the user login form. (Does not require authentication)
    *   POST: Processes user login credentials.
*   **Backend/API Interactions (POST):**
    *   Calls API endpoint `POST /auth/token` to authenticate and retrieve an access token.
    *   Sets `prompthelix_access_token` cookie upon successful login.

#### **Page URL (Route Name):** `/logout (ui_logout)`
*   **HTTP Methods:** GET
*   **Template:** None (Redirects)
*   **Purpose:** Logs out the current user.
*   **Backend/API Interactions:**
    *   Calls API endpoint `POST /auth/logout` to invalidate the session.
    *   Clears the `prompthelix_access_token` cookie.
    *   Redirects to `/login`.

#### **Page URL (Route Name):** `/prompts (list_prompts_ui)`
*   **HTTP Methods:** GET
*   **Template:** `prompts.html`
*   **Purpose:** Displays a list of all prompts. Can show messages or highlight a new version if redirected from another action.
*   **Backend/API Interactions:**
    *   `crud.get_prompts` (delegates to `prompt_service.get_prompts`)

#### **Page URL (Route Name):** `/prompts/new (create_prompt_ui_form)`
*   **HTTP Methods:** GET, POST
*   **Template:** `create_prompt.html`
*   **Purpose:**
    *   GET: Displays the form to create a new prompt and its initial version.
    *   POST: Submits the new prompt and initial version data.
*   **Backend/API Interactions (POST):**
    *   `crud.create_prompt` (delegates to `prompt_service.create_prompt`)
    *   `crud.create_prompt_version` (delegates to `prompt_service.create_prompt_version`)
    *   Redirects to the view prompt page (`/prompts/{prompt_id}`).

#### **Page URL (Route Name):** `/prompts/{prompt_id} (view_prompt_ui)`
*   **HTTP Methods:** GET
*   **Template:** `prompt_detail.html`
*   **Purpose:** Displays the details of a specific prompt, including all its versions (sorted by version number). Can highlight a new version.
*   **Backend/API Interactions:**
    *   `crud.get_prompt` (delegates to `prompt_service.get_prompt`)

#### **Page URL (Route Name):** `/experiments/new (run_experiment_ui_form)`
*   **HTTP Methods:** GET, POST
*   **Template:** `experiment.html`
*   **Purpose:**
    *   GET: Displays the form to configure and run a new Genetic Algorithm (GA) experiment.
    *   POST: Submits the GA experiment parameters.
*   **Backend/API Interactions:**
    *   GET: `crud.get_prompts` (to populate parent prompt dropdown).
    *   POST: Calls API endpoint `POST /api/experiments/run-ga` with `schemas.GAExperimentParams`.
    *   Redirects to dashboard or prompt detail page based on API response.

#### **Page URL (Route Name):** `/playground (llm_playground_ui)`
*   **HTTP Methods:** GET
*   **Template:** `llm_playground.html`
*   **Purpose:** Provides an interface to test prompts with various LLM services. Allows selecting available prompts or entering custom text.
*   **Backend/API Interactions:**
    *   `crud.get_prompts` (to populate available prompts dropdown).
    *   Dynamically fetches prompt content via JavaScript (likely using data embedded in the page or a separate API call not explicitly defined in `ui_routes.py` for content fetching on selection).
    *   Client-side JavaScript will call `POST /api/llm/test_prompt` when "Test Prompt" is clicked.

#### **Page URL (Route Name):** `/settings (view_settings_ui)` or `/settings/ (view_settings_ui_alt)`
*   **HTTP Methods:** GET
*   **Template:** `settings.html`
*   **Purpose:** Displays application settings, including API key configuration for LLM services and available agent information.
*   **Backend/API Interactions:**
    *   `crud.get_api_key` for each supported LLM service.
    *   `list_available_agents()` (local utility function to discover agent classes).

#### **Page URL (Route Name):** `/settings/api_keys (save_api_keys_settings)`
*   **HTTP Methods:** POST
*   **Template:** None (Redirects)
*   **Purpose:** Saves the API key configurations submitted from the settings page.
*   **Backend/API Interactions:**
    *   `crud.create_or_update_api_key` for each submitted API key.
    *   Redirects back to `/settings`.

#### **Page URL (Route Name):** `/admin/users/new (admin_create_user_form)`
*   **HTTP Methods:** GET, POST
*   **Template:** `create_user.html`
*   **Purpose:**
    *   GET: Displays a form for administrators to create new users.
    *   POST: Submits the new user data.
*   **Backend/API Interactions (POST):**
    *   `user_service.get_user_by_username` (for validation)
    *   `user_service.get_user_by_email` (for validation)
    *   `user_service.create_user`
    *   Redirects back to `/admin/users/new` with a message.

#### **Page URL (Route Name):** `/conversations (ui_list_conversations)`
*   **HTTP Methods:** GET
*   **Template:** `conversations.html`
*   **Purpose:** Serves the UI page for viewing conversation logs. The page itself likely uses client-side JavaScript to call API endpoints to fetch and display conversation data.
*   **Backend/API Interactions (client-side):**
    *   Likely calls `GET /api/v1/conversations/sessions/` to list sessions.
    *   Likely calls `GET /api/v1/conversations/sessions/{session_id}/messages/` to get messages for a selected session.

#### **Page URL (Route Name):** `/dashboard (ui_dashboard)`
*   **HTTP Methods:** GET
*   **Template:** `dashboard.html`
*   **Purpose:** Serves the UI page for real-time monitoring, likely for GA experiments or other system metrics. The page uses WebSockets or client-side polling to update.
*   **Backend/API Interactions (client-side/WebSockets):**
    *   Connects to WebSocket endpoints (e.g., `/ws/ga_status`, `/ws/ga_progress_stream`) for real-time updates.
    *   May call `GET /api/ga/status` or other status/metric APIs.

#### **Page URL (Route Name):** `/tests (ui_list_tests)`
*   **HTTP Methods:** GET
*   **Template:** `interactive_tests.html`
*   **Purpose:** Displays a list of available interactive tests.
*   **Backend/API Interactions:**
    *   `list_interactive_tests()` (local utility function).

#### **Page URL (Route Name):** `/tests/run (ui_run_test)`
*   **HTTP Methods:** POST
*   **Template:** `interactive_tests.html` (re-renders)
*   **Purpose:** Runs a selected interactive test and displays its output.
*   **Backend/API Interactions:**
    *   `run_unittest()` (local utility function that uses `unittest` module).
    *   `list_interactive_tests()` (to re-populate the list).

---
*Note: Some UI routes have aliases ending in `.html` which redirect to the primary named route (e.g., `/prompts.html` redirects to `/prompts`). These are not documented separately.*

## Backend Services Documentation

This section describes the core backend services and their functions.

### `prompthelix.services.prompt_service.PromptService`

Handles CRUD operations for Prompts and PromptVersions.

*   **Function:** `PromptService.create_prompt(db, prompt_create, owner_id)`
    *   **Purpose:** Creates a new prompt.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_create`: `schemas.PromptCreate` Pydantic model containing prompt data (name, description).
        *   `owner_id`: Integer ID of the user creating the prompt.
    *   **Returns:** The created `models.Prompt` SQLAlchemy object.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.get_prompt(db, prompt_id)`
    *   **Purpose:** Retrieves a specific prompt by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_id`: Integer ID of the prompt to retrieve.
    *   **Returns:** An optional `models.Prompt` SQLAlchemy object if found, else `None`.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.get_prompts(db, skip=0, limit=100)`
    *   **Purpose:** Retrieves a list of prompts with pagination.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `skip`: Integer, number of records to skip (for pagination).
        *   `limit`: Integer, maximum number of records to return (for pagination).
    *   **Returns:** A list of `models.Prompt` SQLAlchemy objects.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.update_prompt(db, prompt_id, prompt_update)`
    *   **Purpose:** Updates an existing prompt.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_id`: Integer ID of the prompt to update.
        *   `prompt_update`: `schemas.PromptUpdate` Pydantic model with fields to update (name, description).
    *   **Returns:** The updated `models.Prompt` SQLAlchemy object if found and updated, else `None`.
    *   **Notes:** Interacts directly with the database. Fetches the prompt first using `get_prompt`.

*   **Function:** `PromptService.delete_prompt(db, prompt_id)`
    *   **Purpose:** Deletes a prompt by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_id`: Integer ID of the prompt to delete.
    *   **Returns:** The deleted `models.Prompt` SQLAlchemy object if found and deleted, else `None`.
    *   **Notes:** Interacts directly with the database. Also deletes associated prompt versions due to cascading deletes in the database schema (implicitly).

*   **Function:** `PromptService.create_prompt_version(db, prompt_id, version_create)`
    *   **Purpose:** Creates a new version for an existing prompt.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_id`: Integer ID of the parent prompt.
        *   `version_create`: `schemas.PromptVersionCreate` Pydantic model (content, parameters_used, fitness_score).
    *   **Returns:** The created `models.PromptVersion` SQLAlchemy object, or `None` if parent prompt not found.
    *   **Notes:** Interacts directly with the database. Automatically calculates the next `version_number`.

*   **Function:** `PromptService.get_prompt_version(db, prompt_version_id)`
    *   **Purpose:** Retrieves a specific prompt version by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_version_id`: Integer ID of the prompt version.
    *   **Returns:** An optional `models.PromptVersion` SQLAlchemy object if found, else `None`.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.get_prompt_versions_for_prompt(db, prompt_id, skip=0, limit=100)`
    *   **Purpose:** Retrieves all versions for a specific prompt with pagination.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_id`: Integer ID of the parent prompt.
        *   `skip`, `limit`: Integers for pagination.
    *   **Returns:** A list of `models.PromptVersion` SQLAlchemy objects.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.update_prompt_version(db, prompt_version_id, version_update)`
    *   **Purpose:** Updates an existing prompt version.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_version_id`: Integer ID of the prompt version to update.
        *   `version_update`: `schemas.PromptVersionUpdate` Pydantic model with fields to update.
    *   **Returns:** The updated `models.PromptVersion` SQLAlchemy object if found, else `None`.
    *   **Notes:** Interacts directly with the database.

*   **Function:** `PromptService.delete_prompt_version(db, prompt_version_id)`
    *   **Purpose:** Deletes a specific prompt version.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `prompt_version_id`: Integer ID of the prompt version to delete.
    *   **Returns:** The deleted `models.PromptVersion` SQLAlchemy object if found, else `None`.
    *   **Notes:** Interacts directly with the database.

### `prompthelix.services.conversation_service.ConversationService`

Handles retrieval of conversation logs. (Note: Logging of conversations seems to happen elsewhere, possibly in `prompthelix.utils.llm_utils.py` or agent interactions, this service is for reading).

*   **Function:** `ConversationService.get_conversation_sessions(db, skip=0, limit=100)`
    *   **Purpose:** Retrieves a list of unique conversation sessions, including message counts and timestamps for first/last messages.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `skip`, `limit`: Integers for pagination.
    *   **Returns:** A list of `schemas.ConversationSession` Pydantic models.
    *   **Notes:** Interacts directly with the database (`ConversationLog` table). Groups by `session_id`.

*   **Function:** `ConversationService.get_messages_by_session_id(db, session_id, skip=0, limit=1000)`
    *   **Purpose:** Retrieves all messages for a given session ID, ordered by timestamp.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `session_id`: String, the ID of the session.
        *   `skip`, `limit`: Integers for pagination.
    *   **Returns:** A list of `schemas.ConversationLogEntry` Pydantic models (which are representations of `models.ConversationLog`).
    *   **Notes:** Interacts directly with the database.

*   **Function:** `ConversationService.get_all_logs(db, skip=0, limit=100)`
    *   **Purpose:** Retrieves all conversation logs across all sessions, ordered by timestamp descending.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession` object.
        *   `skip`, `limit`: Integers for pagination.
    *   **Returns:** A list of `schemas.ConversationLogEntry` Pydantic models.
    *   **Notes:** Interacts directly with the database.

### `prompthelix.services.user_service`

Handles user creation, retrieval, authentication, and session management.

*   **Function:** `user_service.create_user(db, user_create)`
    *   **Purpose:** Creates a new user.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `user_create`: `schemas.UserCreate` (username, email, password).
    *   **Returns:** The created `models.User` object.
    *   **Notes:** Hashes the password using `pwd_context`. Interacts with the database.

*   **Function:** `user_service.get_user(db, user_id)`
    *   **Purpose:** Retrieves a user by their ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `user_id`: Integer ID of the user.
    *   **Returns:** Optional `models.User` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `user_service.get_user_by_username(db, username)`
    *   **Purpose:** Retrieves a user by their username.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `username`: String username.
    *   **Returns:** Optional `models.User` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `user_service.get_user_by_email(db, email)`
    *   **Purpose:** Retrieves a user by their email.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `email`: String email address.
    *   **Returns:** Optional `models.User` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `user_service.verify_password(plain_password, hashed_password)`
    *   **Purpose:** Verifies a plain password against a stored hashed password.
    *   **Parameters:**
        *   `plain_password`: String.
        *   `hashed_password`: String (the stored hash).
    *   **Returns:** Boolean, `True` if passwords match, `False` otherwise.
    *   **Notes:** Uses `pwd_context.verify`.

*   **Function:** `user_service.update_user(db, user_id, user_update)`
    *   **Purpose:** Updates user information (email, password).
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `user_id`: Integer ID of the user to update.
        *   `user_update`: `schemas.UserUpdate` (optional email, optional password).
    *   **Returns:** Optional updated `models.User` object.
    *   **Notes:** Hashes new password if provided. Interacts with the database.

*   **Function:** `user_service.create_session(db, user_id, expires_delta_minutes=60)`
    *   **Purpose:** Creates a new session for a user.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `user_id`: Integer ID of the user.
        *   `expires_delta_minutes`: Integer, session duration.
    *   **Returns:** The created `models.Session` object.
    *   **Notes:** Generates a secure session token. Interacts with the database.

*   **Function:** `user_service.get_session_by_token(db, session_token)`
    *   **Purpose:** Retrieves a session by its token. Deletes session if expired.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `session_token`: String session token.
    *   **Returns:** Optional `models.Session` object if valid and not expired.
    *   **Notes:** Interacts with the database.

*   **Function:** `user_service.delete_session(db, session_token)`
    *   **Purpose:** Deletes a session by its token.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `session_token`: String session token.
    *   **Returns:** Boolean, `True` if session was found and deleted.
    *   **Notes:** Interacts with the database.

*   **Function:** `user_service.delete_all_user_sessions(db, user_id)`
    *   **Purpose:** Deletes all active sessions for a given user.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `user_id`: Integer ID of the user.
    *   **Returns:** Integer, number of sessions deleted.
    *   **Notes:** Interacts with the database.

### `prompthelix.services.evolution_service`

Manages data related to Genetic Algorithm (GA) experiments, including runs, chromosomes, and generation metrics.

*   **Function:** `evolution_service.create_experiment_run(db, parameters=None)`
    *   **Purpose:** Creates a new record for a GA experiment run.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `parameters`: Optional dictionary of experiment parameters.
    *   **Returns:** The created `models.GAExperimentRun` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.complete_experiment_run(db, run, prompt_version_id=None)`
    *   **Purpose:** Marks a GA experiment run as completed and optionally links it to a resulting prompt version.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run`: The `models.GAExperimentRun` object to update.
        *   `prompt_version_id`: Optional integer ID of the `PromptVersion` created from this run.
    *   **Returns:** The updated `models.GAExperimentRun` object.
    *   **Notes:** Interacts with the database. Sets `completed_at` timestamp.

*   **Function:** `evolution_service.add_chromosome_record(db, run, generation_number, chromosome)`
    *   **Purpose:** Adds a record of a GA chromosome to the database.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run`: The `models.GAExperimentRun` object this chromosome belongs to.
        *   `generation_number`: Integer, the generation number.
        *   `chromosome`: `genetics.PromptChromosome` object.
    *   **Returns:** The created `models.GAChromosome` database object.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.add_generation_metrics(db, run, metrics)`
    *   **Purpose:** Adds a record of GA generation metrics. (Seems to be a more general version of `add_generation_metric`).
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run`: The `models.GAExperimentRun` object.
        *   `metrics`: Dictionary containing metric data (generation_number, best_fitness, avg_fitness, etc.).
    *   **Returns:** The created `models.GAGenerationMetrics` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.get_chromosomes_for_run(db, run_id)`
    *   **Purpose:** Retrieves all chromosome records for a specific GA experiment run.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run_id`: Integer ID of the `GAExperimentRun`.
    *   **Returns:** List of `models.GAChromosome` objects.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.get_experiment_runs(db, skip=0, limit=100)`
    *   **Purpose:** Retrieves a paginated list of GA experiment runs.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `skip`, `limit`: Integers for pagination.
    *   **Returns:** List of `models.GAExperimentRun` objects.
    *   **Notes:** Interacts with the database. Ordered by creation time descending.

*   **Function:** `evolution_service.get_experiment_run(db, run_id)`
    *   **Purpose:** Retrieves a single GA experiment run by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run_id`: Integer ID of the `GAExperimentRun`.
    *   **Returns:** Optional `models.GAExperimentRun` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.add_generation_metric(db, run, generation_number, best_fitness, avg_fitness, population_diversity)`
    *   **Purpose:** Adds specific GA generation metrics (best fitness, average fitness, diversity).
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run`: The `models.GAExperimentRun` object.
        *   `generation_number`: Integer.
        *   `best_fitness`, `avg_fitness`, `population_diversity`: Floats.
    *   **Returns:** The created `models.GAGenerationMetrics` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `evolution_service.get_generation_metrics_for_run(db, run_id)`
    *   **Purpose:** Retrieves all generation metrics for a specific GA run, ordered by generation number.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `run_id`: Integer ID of the `GAExperimentRun`.
    *   **Returns:** List of `models.GAGenerationMetrics` objects.
    *   **Notes:** Interacts with the database.

### `prompthelix.services.performance_service`

Manages performance metrics associated with prompt versions.

*   **Function:** `performance_service.record_performance_metric(db, metric_create)`
    *   **Purpose:** Creates and saves a new performance metric for a prompt version.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `metric_create`: `schemas.PerformanceMetricCreate` (prompt_version_id, metric_name, metric_value).
    *   **Returns:** The created `models.PerformanceMetric` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `performance_service.get_metrics_for_prompt_version(db, prompt_version_id)`
    *   **Purpose:** Retrieves all performance metrics for a specific prompt version.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `prompt_version_id`: Integer ID of the `PromptVersion`.
    *   **Returns:** List of `models.PerformanceMetric` objects.
    *   **Notes:** Interacts with the database.

*   **Function:** `performance_service.get_performance_metric(db, metric_id)`
    *   **Purpose:** Retrieves a specific performance metric by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `metric_id`: Integer ID of the `PerformanceMetric`.
    *   **Returns:** Optional `models.PerformanceMetric` object.
    *   **Notes:** Interacts with the database.

*   **Function:** `performance_service.delete_performance_metric(db, metric_id)`
    *   **Purpose:** Deletes a performance metric by its ID.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `metric_id`: Integer ID of the `PerformanceMetric`.
    *   **Returns:** Boolean, `True` if deleted successfully.
    *   **Notes:** Interacts with the database.

*   **Function:** `performance_service.update_performance_metric(db, metric_id, metric_update)`
    *   **Purpose:** Updates an existing performance metric.
    *   **Parameters:**
        *   `db`: SQLAlchemy `DbSession`.
        *   `metric_id`: Integer ID of the `PerformanceMetric`.
        *   `metric_update`: `schemas.PerformanceMetricUpdate` (optional metric_name, optional metric_value).
    *   **Returns:** Optional updated `models.PerformanceMetric` object.
    *   **Notes:** Interacts with the database.

### `prompthelix.utils.llm_utils`

Utility functions for interacting with Large Language Models (LLMs).

*   **Function:** `llm_utils.list_available_llms(db=None)`
    *   **Purpose:** Returns a list of LLM service names for which API keys are configured.
    *   **Parameters:**
        *   `db`: Optional SQLAlchemy `DbSession` (used if API keys are stored in DB via `crud.get_api_key`).
    *   **Returns:** List of strings (e.g., `["OPENAI", "ANTHROPIC"]`).
    *   **Notes:** Checks configuration (either via `config.get_..._api_key` functions or `crud.get_api_key`).

*   **Function:** `llm_utils.call_openai_api(prompt, model="gpt-3.5-turbo", db=None)`
    *   **Purpose:** Calls the OpenAI API (chat completions).
    *   **Parameters:**
        *   `prompt`: String, the user prompt.
        *   `model`: String, the OpenAI model to use.
        *   `db`: Optional SQLAlchemy `DbSession` (for API key retrieval).
    *   **Returns:** String, the LLM response content or an error string (e.g., "RATE_LIMIT_ERROR").
    *   **Notes:** Retrieves API key using `config.get_openai_api_key`. Handles various OpenAI specific exceptions.

*   **Function:** `llm_utils.call_claude_api(prompt, model="claude-2", db=None)`
    *   **Purpose:** Calls the Anthropic Claude API.
    *   **Parameters:**
        *   `prompt`: String, the user prompt.
        *   `model`: String, the Anthropic model to use.
        *   `db`: Optional SQLAlchemy `DbSession` (for API key retrieval).
    *   **Returns:** String, the LLM response content or an error string.
    *   **Notes:** Retrieves API key using `config.get_anthropic_api_key`. Handles various Anthropic specific exceptions.

*   **Function:** `llm_utils.call_google_api(prompt, model="gemini-pro", db=None)`
    *   **Purpose:** Calls the Google Generative AI (Gemini) API.
    *   **Parameters:**
        *   `prompt`: String, the user prompt.
        *   `model`: String, the Google model to use.
        *   `db`: Optional SQLAlchemy `DbSession` (for API key retrieval).
    *   **Returns:** String, the LLM response content or an error string (e.g., "BLOCKED_PROMPT_ERROR").
    *   **Notes:** Retrieves API key using `config.get_google_api_key`. Handles various Google specific exceptions.

*   **Function:** `llm_utils.call_llm_api(prompt, provider, model=None, db=None)`
    *   **Purpose:** A generic wrapper to call the appropriate LLM API based on the provider.
    *   **Parameters:**
        *   `prompt`: String, the user prompt.
        *   `provider`: String, the LLM provider name (e.g., "openai", "anthropic", "google").
        *   `model`: Optional string, the specific model to use. If `None`, a default for the provider is used.
        *   `db`: Optional SQLAlchemy `DbSession` (passed to provider-specific functions).
    *   **Returns:** String, the LLM response or an error string.
    *   **Notes:** Logs the API call (prompt, provider, model, timestamp, and error if any) to `llm_api_calls.log`. Delegates to provider-specific functions (`call_openai_api`, etc.). This function is also responsible for incrementing LLM usage statistics via `crud.increment_llm_statistic` in the API layer (`api/routes.py`) after this function returns.
