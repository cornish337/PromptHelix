
# PromptHelix

A Python framework for AI prompt generation and optimization using a Prompt DNA System.

This project aims to provide a comprehensive toolkit for developing, evolving, and managing
AI prompts through innovative techniques inspired by genetic algorithms and multi-agent systems.

## Features

*   Genetic Algorithm for prompt optimization
*   Multi-agent system for collaborative prompt engineering
*   Integration with various LLMs (OpenAI, Anthropic, Google)
*   **Conversation Log UI**:
    *   **Purpose**: Allows viewing of logged interactions and messages between agents, the system, and LLMs. This is crucial for debugging, monitoring agent behavior, and understanding communication flows.
    *   **Access**: Available at the `/ui/conversations` route in the web interface.
    *   **Functionality**:
        *   Lists all distinct conversation sessions, showing message counts and time ranges.
        *   Displays messages for a selected session, ordered by timestamp.
        *   Shows sender ID, recipient ID (or "BROADCAST"), message type, full timestamp, and the message content.
        *   Attempts to pretty-print JSON content within messages for readability.
*   **GA Analytics Dashboard**:
    *   **Access**: Visit the `/ui/dashboard` route while the server is running.
    *   **Features**:
        *   Real-time metrics and logs streamed via WebSockets.
        *   Line chart visualizing max, mean, and min fitness across generations.
        *   Agent metrics and conversation events are displayed to show how interactions influence GA evolution.
*   Prometheus metrics exported at `/metrics` for monitoring
*   Performance tracking and evaluation of prompts
*   API for programmatic access and integration
*   User interface for managing and experimenting with prompts (basic HTML interface available)
*   UI pages can be reached with or without a `.html` suffix (e.g., `/ui/tests` and `/ui/tests.html`)

### Inter-Agent Messaging

Agents communicate via an asynchronous message bus. Each agent can subscribe to
specific message types and react when messages are broadcast. For example, the
`ResultsEvaluatorAgent` broadcasts an `evaluation_result` message after scoring a
prompt. The `MetaLearnerAgent` subscribes to both `evaluation_result` and
`critique_result` messages so it can update its knowledge base whenever new
feedback is produced.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following tools installed:

*   **Python 3.9+**: Download from [python.org](https://www.python.org/) or use a version manager.
*   **pip**: Python's package installer, usually comes with Python.
*   **virtualenv**: For creating isolated Python environments. Install using `pip install virtualenv`.
*   **Docker**: (Optional, for Docker-based deployment) Download from [docker.com](https://www.docker.com/).

### Environment Variable Setup

The application requires certain environment variables to be set, especially for connecting to Large Language Model (LLM) APIs and configuring the database. These keys are only necessary if you are running the application in `REAL` mode, which makes actual calls to LLM APIs.

**Required Variables:**

*   `OPENAI_API_KEY`: Your API key for OpenAI services.
*   `ANTHROPIC_API_KEY`: Your API key for Anthropic services.
*   `GOOGLE_API_KEY`: Your API key for Google AI services.
*   `DATABASE_URL`: The connection string for your database (e.g., `postgresql://user:password@host:port/database` for PostgreSQL, or `sqlite:///./prompthelix.db` for SQLite).

*   `WANDB_API_KEY`: Optional key for logging metrics to Weights & Biases.

*   `WANDB_API_KEY` *(optional)*: Enables logging metrics to Weights & Biases when set.
*   `MLFLOW_TRACKING_URI` *(optional)*: URI of your MLflow server for metric logging.

*   `DEBUG`: Set to `true` to enable verbose debug logging.



**Agent Overrides:**

Environment variables can also override default agent settings. Use the pattern `<AGENTNAME>_<SETTING>` where `AGENTNAME` is the agent class name without the `Agent` suffix. Example overrides include:

* `PROMPTARCHITECT_DEFAULT_LLM_MODEL`
* `METALEARNER_PERSIST_KNOWLEDGE_ON_UPDATE`
* `RESULTSEVALUATOR_FITNESS_SCORE_WEIGHTS` (as a JSON string)

**Setting Environment Variables:**

You can set these variables in a few ways:

1.  **Using a `.env` file:**
    Copy the provided `.env.example` to `.env` and fill in your values. The file should look like this:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    DATABASE_URL="sqlite:///./prompthelix.db"
    ```
    The application uses `python-dotenv` to automatically load variables from this file if it exists. Remember to add `.env` to your `.gitignore` file to avoid committing sensitive keys.

2.  **System Environment Variables:**
    You can set these variables directly in your shell or operating system.
    *   On macOS/Linux:
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
        export DATABASE_URL="sqlite:///./prompthelix.db"
        # etc.
        ```
        To make them permanent, add these lines to your shell's configuration file (e.g., `.bashrc`, `.zshrc`).
    *   On Windows:
        Use the System Properties dialog to set environment variables, or use PowerShell:
        ```powershell
        $Env:OPENAI_API_KEY="your_openai_api_key"
        # etc.
        ```

### Setup and Run the Web UI

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL later if known, otherwise leave as placeholder
    cd PromptHelix
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # On macOS and Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # On Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    `setup.py` reads this same file to populate `install_requires`, keeping
    dependency definitions in a single place and reducing drift.
4.  **Initialize the database:**
    When the application first runs (e.g., `uvicorn prompthelix.main:app`), the `init_db()` function is called. For SQLite (the default for development), this will create the database file (e.g., `prompthelix.db`) and all necessary tables if they don't already exist. This is convenient for getting started quickly in a development environment.
    For production environments, and for managing database schema changes over time (migrations) in any environment, PromptHelix uses Alembic. See the "Database Migrations" subsection under "Manual Deployment" for details on using Alembic.
    The SQLite database file is not included in the repository and will be created automatically.
5.  **Run the Web UI (Development Server):**
    ```bash
    uvicorn prompthelix.main:app --reload
    ```
    This will start the Uvicorn server. The `--reload` flag enables auto-reloading when code changes.
6.  **Access the Web UI:**
    Open your web browser and navigate to [http://127.0.0.1:8000/ui/prompts](http://127.0.0.1:8000/ui/prompts).

## Configuration

Application configuration is primarily managed in `prompthelix/config.py`. This file centralizes settings for various components of the application.

Key configuration options include:

*   **LLM Provider Settings**: Configure which LLM providers are enabled and their specific parameters.
*   **API Keys**: API keys for LLM services are loaded from environment variables (as described in the "Environment Variable Setup" section). The `config.py` file handles the retrieval of these keys.
*   **Database Settings**: The database connection URL is also loaded from the `DATABASE_URL` environment variable. `config.py` provides the interface to access this setting.
*   **Application Mode**: Settings for `APP_MODE` (e.g., `TEST`, `REAL`) which can control whether to use mock data or live API calls.

Please refer to the comments and structure within `prompthelix/config.py` for more detailed information on specific configuration options.

### Debugging and Logging

PromptHelix uses a centralized logging configuration defined in `prompthelix/config.py`.
Logging is initialized automatically when running the web app or CLI. To enable
debug-level logs set the environment variable `PROMPTHELIX_DEBUG=1` or pass
`--debug` to the CLI. Logs follow a consistent format with timestamps and module
names. You can also provide a filename via `setup_logging()` in your own scripts
to write logs to a file.

## Deployment

You can deploy PromptHelix using Docker or by setting it up manually in a production environment.

### Docker Deployment

1.  **Build the Docker Image:**
    Navigate to the root directory of the project where the `Dockerfile` is located. Run the following command to build the image:
    ```bash
    docker build -t prompthelix .
    ```
2.  **Run the Docker Container:**
    Once the image is built, you can run it as a container. You'll need to pass your environment variables to the container and map the application port.

    Example `docker run` command:
    ```bash
    docker run -d \
      -p 8000:8000 \
      -e OPENAI_API_KEY="your_openai_api_key" \
      -e ANTHROPIC_API_KEY="your_anthropic_api_key" \
      -e GOOGLE_API_KEY="your_google_api_key" \
      -e DATABASE_URL="postgresql://user:password@host:port/database" \
      --name prompthelix-app \
      prompthelix
    ```
    *   `-d`: Runs the container in detached mode.
    *   `-p 8000:8000`: Maps port 8000 of the host to port 8000 of the container (where Uvicorn runs).
    *   `-e VAR_NAME="value"`: Sets environment variables inside the container.
    *   `--name prompthelix-app`: Assigns a name to the container.

    If you are using a `.env` file and want Docker to use it, you can use the `--env-file` option:
    ```bash
    docker run -d \
      -p 8000:8000 \
      --env-file ./.env \
      --name prompthelix-app \
      prompthelix
    ```
    *Ensure your `.env` file has the correct `DATABASE_URL` for your production database if you use this method.*

#### Docker Compose with Redis

The repository includes a `docker-compose.yaml` that runs both the application and a Redis instance. This is the easiest way to start everything together.

1.  **Copy the example environment file:**
    ```bash
    cp .env.example .env
    ```
    Edit `.env` as needed. When using Docker Compose, set `REDIS_HOST=redis`.

2.  **Build and start the services:**
    ```bash
    docker compose up --build
    ```
    This builds the application image and starts the FastAPI app on port `8000` and Redis on port `6379`.

3.  **Access the application:**
    The API and UI will be available at [http://localhost:8000](http://localhost:8000).


### Manual Deployment

For a manual production setup, consider the following steps:

1.  **Production-Grade ASGI Server:**
    Instead of Uvicorn's development server (`--reload`), use a production-grade ASGI server like Gunicorn or Hypercorn.
    Example with Gunicorn:
    ```bash
    pip install gunicorn
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker prompthelix.main:app -b 0.0.0.0:8000
    ```
    This command starts Gunicorn with 4 worker processes, using Uvicorn workers, and binds to port 8000.

2.  **Reverse Proxy (e.g., Nginx):**
    Set up a reverse proxy like Nginx or Apache in front of your ASGI server. This can handle SSL termination, static file serving, load balancing, and provide an additional layer of security.

3.  **Database Setup (Production):**
    *   For production, it's highly recommended to use a robust database like PostgreSQL.
    *   Set the `DATABASE_URL` environment variable to point to your production database.
    *   **Database Migrations**: PromptHelix uses Alembic for managing database schema changes (migrations). This is crucial for updating your database schema in a controlled way as the application evolves, especially in production.
        *   The `init_db()` function called on application startup is suitable for initial table creation in development (especially with SQLite) but does **not** handle schema migrations.
        *   To apply migrations, use Alembic commands. For example, to upgrade your database to the latest schema version:
            ```bash
            # Ensure your DATABASE_URL environment variable is set correctly for the target database
            alembic upgrade head
            ```
        *   You should run `alembic upgrade head` when deploying new versions of the application to ensure your database schema is up to date.
        *   When creating new models or modifying existing ones, you'll need to generate new migration scripts with Alembic (e.g., `alembic revision -m "create_new_table"` and then edit the script).

4.  **Environment Variables:**
    Ensure all required environment variables (API keys, `DATABASE_URL`, etc.) are securely set in your production environment.

## Project Structure

See `prompthelix/docs/README.md` for a detailed project structure.

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` file for detailed guidelines on how to contribute, report issues, and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Running the MVP

This section describes how to run the Minimum Viable Product functionalities.

### Command Line Interface (CLI)

You can run the genetic algorithm directly from the command line. This will simulate the evolution of prompts and output the best prompt found along with its fitness score at each generation.

Execute the following command from the root of the project:

```bash
python -m prompthelix.cli run ga [options]
```

This command runs the Genetic Algorithm. It supports various options to customize the GA run, including providing an initial seed prompt, setting GA parameters (generations, population size), overriding agent and LLM configurations, specifying an output file for the best prompt, and defining where the population should be persisted.

* `--parallel-workers <integer>`: Number of parallel workers used for fitness evaluation. Set to `1` for serial execution. By default, all available CPU cores are used.
* `--population-file <filepath>`: File path for persisting the GA population. Use this to resume runs or inspect populations. (`--population-path` is a synonym.)

Example:
```bash
python -m prompthelix.cli run ga --parallel-workers 4
```

If you pass a value via the `--prompt` option, the text you supply becomes the first chromosome of the initial generation. This allows you to start the GA from a known prompt rather than generating all prompts randomly.

For a detailed list of all `run ga` options and usage examples, please refer to the [CLI Documentation in `prompthelix/docs/README.md`](prompthelix/docs/README.md#run-command).

### Checking LLM Connectivity

Use the CLI to verify that your API keys are configured correctly and that the selected provider is reachable:

```bash
python -m prompthelix.cli check-llm --provider openai --model gpt-3.5-turbo
```

The command sends a short test prompt and prints the returned text or any error message. Debug logging output is shown to help diagnose connectivity issues.

### Debug Logging

Set the `DEBUG` environment variable to `true` before starting the CLI or server to enable verbose logging:

```bash
export DEBUG=true
python -m prompthelix.cli run ga
```

Modules may also call `setup_logging(json_format=True)` from `prompthelix.utils.logging_utils` to output logs in JSON format.

### API

PromptHelix also provides an API endpoint to trigger the genetic algorithm.

1.  **Start the FastAPI server**:
    Run the Uvicorn server to serve the API:
    ```bash
    uvicorn prompthelix.main:app
    ```

2.  **Access the GA endpoint**:

    Once the server is running, trigger the genetic algorithm by sending a **POST** request to the `/api/experiments/run-ga` endpoint. Example command:
    ```bash
    curl -X POST http://127.0.0.1:8000/api/experiments/run-ga \
         -H "Content-Type: application/json" \
         -d '{"task_description":"Example","keywords":["demo"],"execution_mode":"TEST"}'
    ```


3.  **Try the Prompt Manager UI** *(deprecated)*:
    The original in-memory Prompt Manager has been replaced by the
    database-backed `PromptService`. The existing UI still works but now
    delegates all operations to `PromptService`. See "Setup and Run the
    Web UI" for details.

4.  **Expected Response**:
    The endpoint now launches the experiment asynchronously. It immediately
    returns a `GARunResponse` object with the task identifier and a URL for
    checking progress:

    ```json
    {
      "message": "GA experiment started in background.",
      "task_id": "<uuid>",
      "status_endpoint": "/api/ga/status/<uuid>"
    }
    ```

    The generation results can be monitored via the `status_endpoint` or by
    visiting the GA dashboard at `/ui/dashboard`.

## Metrics and Monitoring

The application exposes Prometheus metrics at `/metrics`. When the optional `wandb` package is installed and `WANDB_API_KEY` is set, these metrics are also logged to Weights & Biases.


### Running Tests

You can run all automated tests (unit and integration tests) using the PromptHelix CLI. By default, the command discovers every test in the `prompthelix/tests` directory and its subdirectories.

Execute the following command from the root of the project:

```bash
python -m prompthelix.cli test
```

To limit discovery to a particular directory or file, pass the `--path` option:

```bash
python -m prompthelix.cli test --path tests/unit/test_architect_agent.py
```

The output will show the progress of the tests and a summary of the results.

### Metrics and Experiment Tracking

PromptHelix exposes Prometheus metrics at the `/metrics` route which report
generation number, best fitness and other GA stats. Configure a Prometheus
server to scrape this endpoint.

If the optional `WANDB_API_KEY` is set, these metrics are logged to Weights &
Biases. Setting `MLFLOW_TRACKING_URI` enables logging to an MLflow server.  Both
integrations activate only when the environment variables are provided.

To execute interactive tests, use the `--interactive` flag. Tests will be discovered under `prompthelix/tests/interactive`:


### Interactive Tests

PromptHelix also provides a simple web interface for running tests manually. Start the
development server and navigate to `/ui/tests` to see the available tests.

1. Open [http://127.0.0.1:8000/ui/tests](http://127.0.0.1:8000/ui/tests) in your browser.
2. Choose a test from the dropdown list and click **Run**.
3. Results will be displayed on the page once execution completes.

If your deployment enforces authentication for UI routes, log in first via
`/ui/login`. Otherwise, the interactive test page can be accessed without a token.

You can also launch the interactive runner from the command line:


```bash
python -m prompthelix.cli test --interactive
```
