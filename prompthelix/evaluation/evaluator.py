# prompthelix/evaluation/evaluator.py

import json
import logging
import openai
from openai import OpenAIError # Use OpenAIError for specific API errors
from prompthelix.config import settings
# Assuming metrics.py is in the same directory
from . import metrics as default_metrics_module

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, metric_functions: list = None, openai_api_key: str = None):
        """
        Initializes the Evaluator.

        Args:
            metric_functions (list, optional): A list of metric functions to use for evaluation.
                                              Defaults to a predefined set from metrics.py.
            openai_api_key (str, optional): OpenAI API key. If None, tries to load from settings.
        """
        if metric_functions is None:
            # Default to some metrics from the metrics module
            self.metric_functions = [
                default_metrics_module.calculate_exact_match,
                default_metrics_module.calculate_keyword_overlap,
                default_metrics_module.calculate_output_length,
                default_metrics_module.calculate_bleu_score # Placeholder
            ]
        else:
            self.metric_functions = metric_functions

        self.evaluation_data = [] # List of {"prompt": str, "expected_output": str}
        self.results = {} # To store detailed results per item

        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in settings or provided. LLM calls will fail.")
            self.openai_client = None
        else:
            try:
                self.openai_client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully for Evaluator.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client for Evaluator: {e}", exc_info=True)
                self.openai_client = None

    def _call_llm_api(self, prompt_string: str, model_name: str = "gpt-3.5-turbo") -> str:
        """
        Calls the LLM API with the given prompt string.
        Args:
            prompt_string (str): The prompt to send to the LLM.
            model_name (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        Returns:
            str: The LLM's response content, or an error message string if the call fails.
        """
        if not self.openai_client:
            logger.error("OpenAI client is not initialized in Evaluator. Cannot call LLM API.")
            return "Error: LLM client not initialized."

        logger.info(f"Evaluator calling OpenAI API model {model_name} for prompt (first 100 chars): {prompt_string[:100]}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt_string}
                ]
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                logger.info(f"Evaluator OpenAI API call successful. Response (first 100 chars): {content[:100]}...")
                return content
            else:
                logger.warning(f"Evaluator OpenAI API call for prompt '{prompt_string[:50]}...' returned no content or unexpected response structure.")
                return "Error: No content from LLM."
        except OpenAIError as e: # More specific error handling
            logger.error(f"Evaluator OpenAI API error for prompt '{prompt_string[:50]}...': {e}", exc_info=True)
            return f"Error: LLM API call failed. Details: {str(e)}"
        except Exception as e:
            logger.critical(f"Evaluator unexpected error during OpenAI API call for prompt '{prompt_string[:50]}...': {e}", exc_info=True)
            return f"Error: Unexpected issue during LLM API call. Details: {str(e)}"

    def load_evaluation_data(self, data_path: str):
        """
        Loads evaluation data from a JSON file.
        The JSON file should contain a list of objects,
        each with "prompt" and "expected_output" keys.
        Example: [{"prompt": "Hello", "expected_output": "Hi there"}]

        Args:
            data_path (str): Path to the JSON file.
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.evaluation_data = json.load(f)
            logger.info(f"Successfully loaded {len(self.evaluation_data)} items from {data_path}")
            # Basic validation of data structure
            if not self.evaluation_data: # Handle empty list case
                logger.warning(f"Evaluation data file {data_path} is empty or contains an empty list.")
                return # No further validation needed for an empty list

            if not isinstance(self.evaluation_data, list):
                 raise ValueError("Data loaded is not a list.")

            if self.evaluation_data and not isinstance(self.evaluation_data[0], dict):
                 raise ValueError("Data items should be dictionaries.")
            # Ensure keys exist in the first item if data is not empty
            if self.evaluation_data and ("prompt" not in self.evaluation_data[0] or "expected_output" not in self.evaluation_data[0]):
                 raise ValueError("Data items must contain 'prompt' and 'expected_output' keys.")
        except FileNotFoundError:
            logger.error(f"Evaluation data file not found: {data_path}")
            self.evaluation_data = []
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {data_path}")
            self.evaluation_data = []
            raise
        except ValueError as ve: # Catch specific validation errors
            logger.error(f"Invalid data format in {data_path}: {ve}")
            self.evaluation_data = []
            raise
        except Exception as e:
            logger.error(f"An error occurred loading evaluation data from {data_path}: {e}", exc_info=True)
            self.evaluation_data = []
            raise

    def run_evaluation(self, model_name: str = "gpt-3.5-turbo"):
        """
        Runs the evaluation process using the loaded data and configured metrics.
        Prompts are executed using the internal LLM API call mechanism.

        Args:
            model_name (str, optional): The LLM model to use for generating outputs.
                                        Defaults to "gpt-3.5-turbo".
        """
        if not self.evaluation_data:
            logger.error("Evaluation data not loaded or is empty. Call load_evaluation_data() with valid data first.")
            # It might be better to return an empty dict or specific status if data is just empty
            # For now, raising an error if it's not loaded at all.
            if not hasattr(self, 'evaluation_data') or self.evaluation_data is None:
                 raise ValueError("Evaluation data not loaded. Call load_evaluation_data() first.")
            elif not self.evaluation_data: # If it's an empty list
                 logger.warning("Evaluation data is empty. No evaluation to run.")
                 self.results = {}
                 return self.results


        if not self.metric_functions:
            logger.error("No metric functions defined for evaluation.")
            raise ValueError("No metric functions defined for evaluation.")

        self.results = {} # Clear previous results
        for i, item in enumerate(self.evaluation_data):
            prompt_text = item.get("prompt")
            # expected_output can be None, and metrics should handle this
            expected_output = item.get("expected_output")

            if not prompt_text:
                logger.warning(f"Skipping item {i} due to missing 'prompt'.")
                self.results[f"item_{i}"] = {
                    "prompt": None, "expected_output": expected_output, "actual_output": None,
                    "scores": {}, "errors": ["Skipped due to missing prompt"]
                }
                continue

            actual_output = self._call_llm_api(prompt_text, model_name=model_name)

            result_item = {
                "prompt": prompt_text,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "scores": {},
                "errors": [] # To log errors during metric calculation for this item
            }

            if actual_output.startswith("Error:"):
                 result_item["errors"].append(f"LLM execution failed: {actual_output}")

            for metric_func in self.metric_functions:
                metric_name = metric_func.__name__
                try:
                    # Pass actual_output and expected_output to the metric function
                    score = metric_func(actual_output, expected_output)
                    result_item["scores"][metric_name] = score
                except Exception as e:
                    logger.error(f"Error calculating metric {metric_name} for item {i} ('{prompt_text[:30]}...'): {e}", exc_info=True)
                    result_item["scores"][metric_name] = None # Indicate error for this metric
                    result_item["errors"].append(f"Metric {metric_name} calculation error: {str(e)}")

            self.results[f"item_{i}"] = result_item

        logger.info(f"Evaluation run completed for {len(self.evaluation_data)} items.")
        return self.results

    def get_results(self):
        """
        Returns the stored evaluation results.
        """
        return self.results

    def add_metric(self, metric_func):
        """
        Adds a new metric function to the evaluator.
        """
        if not callable(metric_func):
            logger.error(f"Attempted to add a non-callable metric: {metric_func}")
            raise ValueError("Metric must be a callable function.")
        if metric_func not in self.metric_functions:
            self.metric_functions.append(metric_func)
            logger.info(f"Metric {metric_func.__name__} added to evaluator.")
        else:
            logger.info(f"Metric {metric_func.__name__} already present in evaluator.")

if __name__ == '__main__':

    # Create a dummy evaluation data file for the example
    dummy_data = [
        {"prompt": "What is the capital of France?", "expected_output": "Paris"},
        {"prompt": "Translate 'hello' to Spanish.", "expected_output": "Hola"},
        {"prompt": "Who wrote 'Hamlet'?", "expected_output": "William Shakespeare"},
        {"prompt": "This prompt will cause an LLM error if API key is invalid or not set.", "expected_output": "Any"}
    ]
    dummy_data_path = "prompthelix_eval_data.json" # Created in the current working directory
    try:
        with open(dummy_data_path, 'w') as f:
            json.dump(dummy_data, f)
        logger.info(f"Dummy evaluation data written to {dummy_data_path}")
    except IOError as e:
        logger.error(f"Failed to write dummy data file {dummy_data_path}: {e}", exc_info=True)
        exit(1) # Exit if we can't even create the dummy file

    # Ensure OPENAI_API_KEY is set in your environment or settings for this to run fully
    # For testing without a key, it will show LLM call errors.
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set in settings. LLM calls in the example will likely fail and return error messages.")
        logger.warning("Metrics will be calculated on these error messages as 'actual_output'.")

    # Initialize Evaluator (it will use default metrics)
    evaluator = Evaluator()

    # Load evaluation data
    try:
        evaluator.load_evaluation_data(dummy_data_path)
        logger.info(f"Evaluation data loaded: {len(evaluator.evaluation_data)} items.")
    except Exception as e:
        logger.error(f"Failed to load evaluation data for example: {e}", exc_info=True)
        # Exit if data loading fails, as run_evaluation depends on it
        # os.remove(dummy_data_path) # Clean up before exiting
        exit(1)

    # Run evaluation (using the default model "gpt-3.5-turbo")
    # If OPENAI_API_KEY is not set, actual_output will be an error string.
    try:
        results = evaluator.run_evaluation()
        logger.info("\n--- Evaluation Results ---")
        if results:
            for item_id, result_details in results.items():
                print(f"  Item ID: {item_id}")
                print(f"    Prompt: {result_details['prompt']}")
                print(f"    Expected Output: {result_details['expected_output']}")
                print(f"    Actual Output: {result_details['actual_output']}")
                print(f"    Scores: {result_details['scores']}")
                if result_details.get('errors'): # Use .get for safety
                    print(f"    Item Errors: {result_details['errors']}")
                print("-" * 20)
        else:
            logger.info("No results to display from evaluation run.")

    except ValueError as ve: # Catch ValueErrors from run_evaluation (e.g. no data/metrics)
         logger.error(f"ValueError during evaluation run: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation run: {e}", exc_info=True)

    # Example of adding a custom metric (if needed)
    # def custom_word_count(output, expected):
    #     if output is None: return 0
    #     return len(output.split())
    # evaluator.add_metric(custom_word_count)
    # logger.info("Added custom metric and could re-run evaluation if desired.")

    # Clean up dummy file (optional)
    # import os
    # try:
    #     os.remove(dummy_data_path)
    #     logger.info(f"Dummy data file {dummy_data_path} removed.")
    # except OSError as e:
    #     logger.error(f"Error removing dummy data file {dummy_data_path}: {e}", exc_info=True)
