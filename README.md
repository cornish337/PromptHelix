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
*   User interface for managing and experimenting with prompts

## Getting Started

(Instructions to be added once the project is more mature)

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
    Run the following command from the root of the project:
    ```bash
    uvicorn prompthelix.main:app --reload
    ```
    This will start the Uvicorn server. The `--reload` flag enables auto-reloading when code changes, which is useful for development.

2.  **Access the GA endpoint**:
    Once the server is running, you can trigger the genetic algorithm by sending a GET request to the `/api/run-ga` endpoint. For example, using `curl`:
    ```bash
    curl http://127.0.0.1:8000/api/run-ga
    ```
    Or you can open `http://127.0.0.1:8000/api/run-ga` in your web browser.

3.  **Expected Response**:
    The API will return a JSON response containing the best prompt found by the genetic algorithm and its fitness score:
    ```json
    {
        "best_prompt": "some generated prompt text",
        "fitness": 20
    }
    ```
    Note: The actual prompt and fitness score will vary with each run due to the nature of the genetic algorithm.
