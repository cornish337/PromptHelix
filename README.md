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

To get PromptHelix up and running on your local machine, follow these steps:

1.  **Prerequisites:**
    *   Python 3.8 or higher.
    *   `pip` for installing packages.
    *   `git` for cloning the repository.

2.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd prompthelix
    ```

3.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Database Setup:**
    The project defines SQLAlchemy models for data persistence (see `prompthelix/models/`). Currently, the core genetic algorithm and API endpoint run primarily in-memory. For future features involving data storage and user accounts, you would typically initialize a database and run migrations. For now, no specific database setup is required to run the MVP features described below.

Now you are ready to run the application's MVP features as described in the "Running the MVP" section.

## Project Structure

See `prompthelix/docs/README.md` for a detailed project structure.

## Contributing

Contributions are welcome! If you'd like to contribute to PromptHelix, please follow these guidelines:

1.  **Reporting Bugs:**
    *   If you find a bug, please open an issue on the project's issue tracker.
    *   Describe the bug in detail, including steps to reproduce it.

2.  **Suggesting Enhancements:**
    *   Open an issue to discuss new features or enhancements.

3.  **Making Changes:**
    *   Fork the repository.
    *   Create a new branch for your feature or bug fix (e.g., `git checkout -b feature/your-feature-name` or `fix/bug-description`).
    *   Make your changes. Adhere to PEP 8 coding style guidelines.
    *   Ensure all tests pass. You can run tests using the command:
        ```bash
        python -m prompthelix.cli test
        ```
    *   Commit your changes with a clear and descriptive commit message.
    *   Push your branch to your fork.
    *   Open a pull request against the main repository.

4.  **Code of Conduct:**
    *   (Optional: Add a link to a Code of Conduct file if one exists, otherwise remove this point or state general expectations of respectful interaction).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

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

### Running Tests

You can run all automated tests (unit and integration tests) using the PromptHelix CLI. This command will discover and execute all tests located within the `prompthelix/tests` directory and its subdirectories.

Execute the following command from the root of the project:

```bash
python -m prompthelix.cli test
```

The output will show the progress of the tests and a summary of the results.

