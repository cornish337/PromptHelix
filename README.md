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
    *Note: Ensure you have Python 3.9+ installed.*
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Initialize the database:**
    The application uses SQLite and will automatically create and initialize the database (`prompthelix.db`) on first run via `init_db()` in `main.py`. For production deployments, you might consider using Alembic for database migrations if you switch to a database like PostgreSQL.
5.  **Run the Web UI:**
    ```bash
    uvicorn prompthelix.main:app --reload
    ```
    This will start the Uvicorn server. The `--reload` flag enables auto-reloading when code changes, which is useful for development.
6.  **Access the Web UI:**
    Open your web browser and navigate to [http://127.0.0.1:8000/ui/prompts](http://127.0.0.1:8000/ui/prompts).

## Project Structure

See `prompthelix/docs/README.md` for a detailed project structure.

## Contributing

(Contribution guidelines to be added)

## License

(License information to be added - likely MIT)

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

