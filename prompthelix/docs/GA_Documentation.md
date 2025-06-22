# PromptHelix Genetic Algorithm: In-Depth Documentation

This document provides a detailed explanation of the Genetic Algorithm (GA) implemented in PromptHelix, including its core components, how agents interact within it, the evaluation process, and areas that may require attention for optimizing prompts against real Large Language Models (LLMs).

## 1. Genetic Algorithm Core Components

The GA is designed to evolve a population of prompts (`PromptChromosome` objects) over generations to improve their effectiveness based on defined fitness criteria.

### 1.1. `PromptChromosome` (`prompthelix/genetics/chromosome.py`)

*   **Structure**: Represents a single prompt in the GA.
    *   `id`: A unique UUID.
    *   `genes`: A list of strings, where each string is a segment or component of the prompt (e.g., instruction, context, formatting rule).
    *   `fitness_score`: A float indicating the prompt's quality, assigned by the evaluation process.
    *   `parent_ids`: A list of IDs of parent chromosomes from which this chromosome might have been derived through crossover.
    *   `mutation_strategy`: A string indicating the name of the mutation strategy applied to create this chromosome, if any.
*   **Role**: These are the individuals in the GA population. They are selected, combined (crossover), and modified (mutation).
*   **Key Methods**:
    *   `clone()`: Creates a deep copy with a new ID.
    *   `to_prompt_string()`: Joins genes into a single string representation of the prompt.

### 1.2. `GeneticOperators` (`prompthelix/genetics/engine.py`)

This class provides the mechanisms for evolving the population.

*   **Selection (`selection(population, tournament_size)`)**:
    *   **Method**: Implements tournament selection. A random subset (tournament) is chosen from the population, and the fittest individual from this subset is selected.
*   **Crossover (`crossover(parent1, parent2, crossover_rate)`)**:
    *   **Method**: Implements single-point crossover. With a probability (`crossover_rate`), two parent chromosomes exchange genetic material at a random pivot point in their gene lists. Children inherit genes from both parents. If no crossover occurs, children are clones of parents (with new IDs and reset fitness). Parent IDs are recorded in the children.
*   **Mutation (`mutate(chromosome, mutation_rate, gene_mutation_prob, target_style)`)**:
    *   **Method**: Modifies a chromosome's genes.
    *   **Process**: A chromosome has a `mutation_rate` chance to undergo mutation. If selected, one strategy is chosen from a list of `mutation_strategies` (provided during initialization, defaults to `NoOperationMutationStrategy`).
    *   **Mutation Strategies**: These are classes responsible for specific gene modifications (e.g., `RandomCharacterAppend`, `GeneSwapMutation` - specific strategies would be defined in `prompthelix/genetics/mutation_strategies.py`). The chosen strategy modifies the chromosome.
    *   **`StyleOptimizerAgent` Integration**: If a `style_optimizer_agent` is provided to `GeneticOperators` and a `target_style` is specified during the `mutate` call, this agent is invoked to potentially refine the chromosome's genes according to the target style. This allows for "smarter," context-aware mutations. The `mutation_strategy` attribute of the chromosome is updated with the name of the applied strategy.

### 1.3. `FitnessEvaluator` (`prompthelix/genetics/engine.py`)

*   **Role**: This class is responsible for initiating the fitness evaluation of a `PromptChromosome`.
*   **Process**:
    1.  Takes a `PromptChromosome`, `task_description`, and optional `success_criteria`. It also uses `llm_settings` passed during its initialization.
    2.  Converts the chromosome's genes into a `prompt_string`.
    3.  **LLM Interaction & Output Generation**:
        *   If `execution_mode` is `ExecutionMode.TEST`:
            *   It generates a `mock_llm_output` string. This output is then passed to the `ResultsEvaluatorAgent`.
        *   If `execution_mode` is *not* `TEST` (e.g., `ExecutionMode.REAL`):
            *   **Synthetic Test Generation (Optional)**: If `llm_settings['num_synthetic_inputs_for_evaluation']` is greater than 0, the `FitnessEvaluator` first calls an LLM (using `prompthelix.utils.llm_utils.call_llm_api` with provider/model from `llm_settings`) to generate a set of diverse input scenarios based on the `task_description`.
                *   For each synthetic input:
                    *   The original `prompt_string` is combined with the synthetic input.
                    *   This combined prompt is sent to the target LLM (configured via `llm_settings`, using `call_llm_api`) to get an actual `llm_output`.
                    *   This `llm_output` is then evaluated by `ResultsEvaluatorAgent` (see below), potentially with the `synthetic_input_context` passed for more contextualized assessment by the agent.
                *   The final fitness score for the chromosome becomes an aggregation (e.g., average) of scores from all synthetic tests.
            *   **Direct LLM Call (if synthetic tests are disabled)**: If synthetic tests are not enabled (`num_synthetic_inputs_for_evaluation` is 0 or not set), the original `prompt_string` is sent directly to the target LLM (configured via `llm_settings`, using `call_llm_api`) to obtain a single `llm_output`.
    4.  **Delegates to `ResultsEvaluatorAgent`**:
        *   If synthetic tests were run, `ResultsEvaluatorAgent.process_request()` is called for *each* synthetic input's LLM output.
        *   If synthetic tests were not run (or in `TEST` mode), `ResultsEvaluatorAgent.process_request()` is called once with the single (real or mock) `llm_output`.
    5.  The fitness score (either aggregated from synthetic tests or from a single evaluation) returned by `ResultsEvaluatorAgent` is then assigned to the `PromptChromosome`.

### 1.4. `PopulationManager` (`prompthelix/genetics/engine.py`)

*   **Role**: Orchestrates the GA lifecycle, managing the population of `PromptChromosome` individuals and driving evolution.
*   **Initialization (`initialize_population`)**:
    *   Uses an instance of `PromptArchitectAgent` to generate the initial population of prompts based on a task description, keywords, and constraints.
    *   Evaluates this initial population using the `FitnessEvaluator`.
*   **Evolutionary Cycle (`evolve_population`)**:
    *   The `evolve_population` method in `PopulationManager` handles the evaluation and sorting phase of a generation.
    *   It iterates through the current population, calling `fitness_evaluator.evaluate()` for each chromosome (which now involves real LLM calls and potential synthetic test evaluations as described above).
    *   **User Feedback Integration**: If a database session is provided to `evolve_population`, after the initial fitness scores are calculated, it queries the database (using `prompthelix.api.crud` functions) for user feedback associated with each chromosome (by ID or content).
        *   User ratings (e.g., 1-5 stars) are used to adjust the fitness scores of the corresponding chromosomes (e.g., higher ratings boost fitness, lower ratings penalize it).
        *   This allows direct human guidance to influence the selection process for the next generation.
    *   The population is then sorted based on these potentially user-feedback-adjusted fitness scores.
    *   The broader GA loop (selection, crossover, mutation, replacement) is typically managed by a runner script (e.g., `ga_runner.py`), which calls `evolve_population` and then applies genetic operators.
*   **Other Features**: Includes methods for pausing, resuming, stopping evolution, saving/loading populations, and broadcasting GA status updates via a message bus.

## 2. Agent Interactions within the GA

The GA leverages several specialized agents:

### 2.1. `PromptArchitectAgent` (`prompthelix/agents/architect.py`)

*   **Role**: Designs the initial set of `PromptChromosome` individuals.
*   **Process**:
    *   Receives a task description, keywords, and constraints.
    *   Uses an LLM (via `call_llm_api`) to:
        1.  Parse and understand the requirements.
        2.  Select an appropriate prompt template from its knowledge base (e.g., `architect_knowledge.json`).
        3.  Populate the genes of a new `PromptChromosome` by filling the template based on the requirements.
    *   Has fallback mechanisms if LLM calls fail.
*   **Outcome**: Produces diverse and contextually relevant initial prompts for the GA.

### 2.2. `ResultsEvaluatorAgent` (`prompthelix/agents/results_evaluator.py`)

*   **Role**: Assesses the quality of an `llm_output` (provided by `FitnessEvaluator`) and calculates a fitness score.
*   **Input**: Receives a request dictionary typically containing:
    *   `prompt_chromosome`: The original `PromptChromosome` being evaluated.
    *   `llm_output`: The actual output generated by an LLM in response to the `prompt_chromosome`'s prompt (or a synthetic variation).
    *   `task_description`: The original task definition.
    *   `success_criteria`: Constraints for the output.
    *   `synthetic_input_context` (optional): If the `llm_output` was generated from a prompt combined with a synthetic input, this field provides that input scenario for more contextualized evaluation.
*   **Process**:
    1.  **Constraint Checking (`_check_constraints`)**: Evaluates the `llm_output` against `success_criteria` (e.g., length, required/forbidden keywords).
    2.  **Content Analysis (`_analyze_content`)**:
        *   It calls its own configured LLM (e.g., `gpt-4`) with a specialized prompt. This prompt asks the LLM to evaluate the received `llm_output` for quality aspects like relevance, coherence, completeness, accuracy, and safety.
        *   The evaluation is performed in the context of the original `task_description`, the `prompt_chromosome` that (partially or wholly) generated the prompt, and crucially, any `synthetic_input_context` provided. This allows the LLM assessor to understand if the output is good *given a specific input scenario*.
        *   Parses the LLM's JSON response to extract these scores.
        *   Has fallback mechanisms if its internal LLM call fails.
    3.  **Fitness Calculation**: Combines scores from constraint checking and the LLM-based content analysis using configurable weights to produce a final `fitness_score`.
*   **Configuration**: Managed through its settings, including LLM provider/model for its internal evaluation, fitness score weights, and path to its knowledge file.

### 2.3. `StyleOptimizerAgent` (e.g., `prompthelix/agents/style_optimizer.py`)

*   **Role**: Intended for "smart" or context-aware mutations of prompts.
*   **Integration**: `GeneticOperators.mutate` can call this agent if it's provided during initialization and a `target_style` is given to the `mutate` function.
*   **Process**: The specific implementation of `StyleOptimizerAgent` would define how it refines a `PromptChromosome` based on a `target_style`. This could involve LLM calls to rephrase, restructure, or add stylistic elements to the prompt's genes.
*   **Status**: The hook for this agent is present in `GeneticOperators`. Its effectiveness depends on its own implementation and how it's utilized in the GA run configuration.

### 2.4. Other Agents

*   **`PromptCriticAgent`**: Potentially used to critique prompts, perhaps by `ResultsEvaluatorAgent` or `PromptArchitectAgent` to refine their outputs or evaluation criteria. Not directly in the main GA loop described in `engine.py`.
*   **`DomainExpertAgent`**: Could provide domain-specific knowledge or evaluation heuristics, possibly consulted by `ResultsEvaluatorAgent`.
*   **`MetaLearnerAgent`**: Designed to consume data from GA runs (e.g., fitness scores, successful prompts) to adapt strategies or configurations over time (e.g., updating agent knowledge files).

## 3. Genetic Algorithm Evaluation Process (Enhanced)

The evaluation of a `PromptChromosome`'s fitness has been significantly enhanced:

1.  A GA runner script (e.g., `ga_runner.py`) iterates through generations. For each chromosome in the current generation:
2.  The `PopulationManager`'s `evolve_population` method calls `FitnessEvaluator.evaluate(chromosome, task_description, success_criteria, llm_settings)`.
3.  **Inside `FitnessEvaluator.evaluate()`**:
    a.  The `chromosome`'s genes are converted to a `prompt_string`.
    b.  **LLM Interaction & Output Generation**:
        *   **TEST Mode**: If `execution_mode == ExecutionMode.TEST`, a `mock_llm_output` is generated.
        *   **REAL Mode (with Synthetic Tests)**: If `execution_mode != ExecutionMode.TEST` and `llm_settings['num_synthetic_inputs_for_evaluation'] > 0`:
            i.  An LLM (via `call_llm_api`) generates multiple `synthetic_input` scenarios based on `task_description`.
            ii. For each `synthetic_input`:
                1.  The `prompt_string` is combined with the `synthetic_input`.
                2.  This combined prompt is sent to the target LLM (via `call_llm_api`) to get an `actual_llm_output`.
                3.  `ResultsEvaluatorAgent.process_request()` is called with the `chromosome`, this `actual_llm_output`, `task_description`, `success_criteria`, and the `synthetic_input` (as context).
            iii. The chromosome's `fitness_score` becomes an average of scores from these synthetic tests.
        *   **REAL Mode (without Synthetic Tests)**: If `execution_mode != ExecutionMode.TEST` and synthetic tests are disabled:
            i.  The `prompt_string` is sent to the target LLM (via `call_llm_api`) to get an `actual_llm_output`.
            ii. `ResultsEvaluatorAgent.process_request()` is called with this `actual_llm_output`.
            iii. The returned score is the chromosome's `fitness_score`.
    c.  The `fitness_score` (single or averaged) is assigned to `chromosome.fitness_score`.
4.  **Inside `ResultsEvaluatorAgent.process_request()`**:
    a.  The agent checks the received `llm_output` (which is now a real LLM output in REAL mode) against `success_criteria`.
    b.  It then calls its own configured LLM. The prompt to this evaluation LLM includes the `task_description`, the original `prompt_chromosome`'s content, the received `llm_output`, and importantly, any `synthetic_input_context` if the output was from a synthetic test. This allows for a more contextualized quality assessment.
    c.  Scores from its internal LLM analysis (e.g., `llm_assessed_quality`) are extracted.
    d.  A final `fitness_score` for that specific output is calculated using configured weights.
5.  **Back in `PopulationManager.evolve_population()`**:
    a.  After all chromosomes have their initial fitness evaluated by `FitnessEvaluator`.
    b.  **User Feedback Integration**: If a `db_session` is provided, the system queries for user feedback (ratings) associated with each chromosome.
    c.  The `fitness_score` of chromosomes is adjusted based on these user ratings (e.g., positive ratings increase fitness, negative ratings decrease it).
    d.  The population is then sorted by these (potentially feedback-adjusted) fitness scores.
6.  The GA proceeds with selection, crossover, and mutation to create the next generation.

## 4. Key Features and Improvements

The GA implementation now incorporates several key features for more effective prompt optimization and user interaction:

*   **Real LLM Output for Fitness Evaluation**: In `REAL` mode, `FitnessEvaluator` now uses `prompthelix.utils.llm_utils.call_llm_api` to send the candidate prompt (or its variations with synthetic inputs) to the actual target LLM. The response from this LLM is then used by `ResultsEvaluatorAgent` for quality assessment. This ensures that prompts are evolved based on their actual performance.
*   **Synthetic Test Generation for Robustness**:
    *   Controlled by the `num_synthetic_inputs_for_evaluation` parameter in `llm_settings` (configurable via CLI: `--num-synthetic-inputs`).
    *   When enabled, for each candidate prompt, `FitnessEvaluator` generates multiple diverse input scenarios using an LLM.
    *   The candidate prompt is then tested against each synthetic input, and its overall fitness is an aggregation of these test outcomes. This helps in evolving prompts that are robust and generalize well across various inputs.
    *   `ResultsEvaluatorAgent` considers the `synthetic_input_context` when performing its LLM-based quality assessment, leading to more accurate evaluations for these varied inputs.
*   **User Feedback Loop**:
    *   A `UserFeedback` data model and API endpoints (`/api/feedback/`) allow users to submit ratings (1-5 stars) and textual feedback on prompts/chromosomes.
    *   During the GA run, `PopulationManager.evolve_population` can query these feedback entries (if a database session is available).
    *   The fitness scores of chromosomes are adjusted based on user ratings, directly influencing the selection process and guiding the evolution towards user-preferred solutions.
*   **Contextualized Evaluation by `ResultsEvaluatorAgent`**: The agent's internal LLM assessment prompt now includes any `synthetic_input_context`, allowing its evaluation LLM to make more informed judgments about the output's quality relative to the specific input scenario it was generated for.

These changes address the previous limitation where the GA was not optimizing against real LLM outputs and lacked mechanisms for dynamic input variation or user guidance.

## 5. Summary of Key Agent Structures

*   **`BaseAgent` (`prompthelix/agents/base.py`)**:
    *   Provides foundational attributes and methods for all agents, such as `agent_id`, `message_bus` integration, `settings` management, and logging.

*   **`PromptArchitectAgent` (`prompthelix/agents/architect.py`)**:
    *   **Purpose**: To design and generate diverse initial `PromptChromosome` objects for the GA's starting population.
    *   **Key Methods/Logic**:
        *   `process_request(request_data)`: Takes task details and uses an LLM to parse requirements, select a suitable prompt template (from its JSON knowledge file, e.g., `architect_knowledge.json`), and populate the template's sections to create genes for a new chromosome.
        *   Handles LLM communication (via `call_llm_api`) and has fallbacks for template selection and gene population if LLM calls fail.

*   **`ResultsEvaluatorAgent` (`prompthelix/agents/results_evaluator.py`)**:
    *   **Purpose**: To evaluate a given LLM output (currently mock/empty from `FitnessEvaluator`) and assign a fitness score to the associated `PromptChromosome`.
    *   **Key Methods/Logic**:
        *   `process_request(request_data)`: Main entry point. Takes the chromosome, the (mock/empty) `llm_output`, task description, and success criteria.
        *   `_check_constraints()`: Checks the `llm_output` against defined constraints (length, keywords).
        *   `_analyze_content()`: Uses its own configured LLM to assess the quality (relevance, coherence, etc.) of the *input `llm_output`*. This is where the evaluation of the mock/empty string happens.
        *   Calculates a final fitness score based on constraint adherence and the LLM-based quality assessment, using configured weights.
        *   Loads its configuration (LLM models, metric weights) from settings and a JSON knowledge file (e.g., `results_evaluator_config.json`).

*   **`StyleOptimizerAgent` (e.g., `prompthelix/agents/style_optimizer.py` - implementation not reviewed here but hook exists):**
    *   **Purpose**: To perform intelligent, context-aware mutations or refinements on `PromptChromosome` objects.
    *   **Integration**: Called by `GeneticOperators.mutate()` if configured and a `target_style` is provided.
    *   **Logic**: Would typically involve using an LLM or rule-based systems to modify prompt genes to align with a desired style, improve clarity, or enhance effectiveness based on learned patterns.


## 6. Conclusion

The PromptHelix GA provides a significantly enhanced framework for prompt evolution. Key improvements include:

*   **Realistic Fitness Evaluation**: Prompts are now evaluated based on their actual output from target LLMs, rather than mock data, especially when `execution_mode` is `REAL`.
*   **Improved Robustness through Synthetic Testing**: The optional synthetic test generation feature allows for evaluating prompts against a diverse set of programmatically generated input scenarios, promoting the evolution of more generalizable and robust prompts. The `ResultsEvaluatorAgent` performs contextualized analysis of these synthetic test outputs.
*   **Human-in-the-Loop Guidance**: The integration of user feedback (ratings) directly influences chromosome fitness, allowing human expertise and preference to guide the evolutionary process.
*   **Flexible Agent Interactions**: The system continues to leverage specialized agents like `PromptArchitectAgent` for initial population generation and `ResultsEvaluatorAgent` for detailed, (now more contextualized) LLM-based quality assessment.

These enhancements make the PromptHelix GA a more powerful and practical tool for optimizing prompts for real-world LLM performance, incorporating both automated rigor and human insight. Future work could involve using the textual suggestions from user feedback to guide mutation strategies more directly or evolving the synthetic test cases themselves (Genetic Critics).