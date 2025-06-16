# PromptHelix

A Python framework for AI prompt generation and optimization using a Prompt DNA System.

This project aims to provide a comprehensive toolkit for developing, evolving, and managing
AI prompts through innovative techniques inspired by genetic algorithms and multi-agent systems.

## Features (Planned)

*   Genetic Algorithm for prompt optimization
*   Multi-agent system for collaborative prompt engineering
*   Integration with various LLMs (OpenAI, Anthropic, Google)
*   Performance tracking and evaluation of prompts
*   API for programmatic access and integration
*   User interface for managing and experimenting with prompts (basic HTML interface available)

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

**Setting Environment Variables:**

You can set these variables in a few ways:

1.  **Using a `.env` file:**
    Create a file named `.env` in the root directory of the project. Add your environment variables to this file in the following format:
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
    cd prompthelix
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
4.  **Initialize the database:**
    If using SQLite (the default for development), the database (`prompthelix.db`) will be automatically created and initialized on the first run via `init_db()` in `main.py`. For production, see the "Deployment" section for database setup and migrations.
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
    *   **Database Migrations**: If you are using a database like PostgreSQL, you will need to manage database schema changes. It's mentioned that the project might consider Alembic. If Alembic is integrated (check `prompthelix/alembic`), you would typically run migrations like this:
        ```bash
        # Ensure ALEMBIC_CONFIG is set or alembic.ini is configured
        cd /path/to/PromptHelix
        alembic upgrade head
        ```
        Run this command whenever deploying to a new environment so your production database schema matches the models. If Alembic is not yet fully set up, you might need to initialize it or use manual SQL scripts for schema management. The current `init_db()` is for SQLite and development.

4.  **Environment Variables:**
    Ensure all required environment variables (API keys, `DATABASE_URL`, etc.) are securely set in your production environment.

## Project Structure

See `prompthelix/docs/README.md` for a detailed project structure.

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` file for detailed guidelines on how to contribute, report issues, and submit pull requests.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Running the MVP

This section describes how to run the Minimum Viable Product functionalities.

### Command Line Interface (CLI)

You can run the genetic algorithm directly from the command line. This will simulate the evolution of prompts and output the best prompt found along with its fitness score at each generation.

Execute the following command from the root of the project:

```bash
python -m prompthelix.cli run
```

This command runs the `run_ga.py` script via the CLI, which prints the progress of the genetic algorithm and the best prompt at the end of the process.

### API

PromptHelix also provides an API endpoint to trigger the genetic algorithm.

1.  **Start the FastAPI server**:
    Ensure the FastAPI server is running as described in the "Setup and Run the Web UI" section.

2.  **Access the GA endpoint**:
    Once the server is running, you can trigger the genetic algorithm by sending a GET request to the `/api/run-ga` endpoint. For example, using `curl`:
    ```bash
    curl http://127.0.0.1:8000/api/run-ga
    ```
    Or you can open `http://127.0.0.1:8000/api/run-ga` in your web browser.

3.  **Try the Prompt Manager UI**:
    The Prompt Manager UI, for adding and viewing prompts, can be accessed as described in the "Setup and Run the Web UI" section.

4.  **Expected Response**:
    The API will return a JSON response containing the best prompt found by the genetic algorithm and its fitness score:
    ```json
    {
        "best_prompt": "some generated prompt text",
        "fitness": 20
    }
    ```
    Note: The actual prompt and fitness score will vary with each run due to the nature of the genetic algorithm.

### Running Tests

You can run all automated tests (unit and integration tests) using the PromptHelix CLI. This command will discover and execute all tests located within the `prompthelix/tests` directory and its subdirectories.

Execute the following command from the root of the project:

```bash
python -m prompthelix.cli test
```

The output will show the progress of the tests and a summary of the results.

