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
    1.  Takes a `PromptChromosome`, `task_description`, and optional `success_criteria`.
    2.  Converts the chromosome's genes into a `prompt_string`.
    3.  **LLM Interaction (Mock/Empty)**:
        *   If `execution_mode` is `ExecutionMode.TEST`: It generates a `mock_llm_output` string containing snippets of the prompt and random numbers.
        *   If `execution_mode` is *not* `TEST`: The `llm_output` remains an **empty string**.
    4.  **Delegates to `ResultsEvaluatorAgent`**: It passes the `PromptChromosome`, the generated `llm_output` (mock or empty), `task_description`, and `success_criteria` to an instance of `ResultsEvaluatorAgent`.
    5.  The fitness score returned by `ResultsEvaluatorAgent` is then assigned to the `PromptChromosome`.
*   **Critical Note**: This component, as currently implemented in `engine.py`, **does not call a real LLM** to get an output from the `prompt_string` it generates from the chromosome.

### 1.4. `PopulationManager` (`prompthelix/genetics/engine.py`)

*   **Role**: Orchestrates the GA lifecycle, managing the population of `PromptChromosome` individuals and driving evolution.
*   **Initialization (`initialize_population`)**:
    *   Uses an instance of `PromptArchitectAgent` to generate the initial population of prompts based on a task description, keywords, and constraints.
    *   Evaluates this initial population using the `FitnessEvaluator`.
*   **Evolutionary Cycle (`evolve_population`)**:
    *   The `evolve_population` method in `PopulationManager` currently focuses on the **evaluation and sorting phase** of a generation. It iterates through the current population, calls `fitness_evaluator.evaluate()` for each chromosome, and then sorts the population by fitness.
    *   The broader GA loop (selection, crossover, mutation, replacement) is typically managed by a runner script (e.g., `ga_runner.py` or `run_ga.py`), which would call `evolve_population` as part of each generation's processing, followed by applying genetic operators using `GeneticOperators` to create the next generation.
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
*   **Input**: Receives:
    *   `prompt_chromosome`: The original chromosome being evaluated.
    *   `llm_output`: **Critically, this is the mock or empty string from `FitnessEvaluator`**.
    *   `task_description`: The original task.
    *   `success_criteria`: Constraints for the output.
*   **Process**:
    1.  **Constraint Checking (`_check_constraints`)**: Evaluates the `llm_output` against `success_criteria` (e.g., length, required/forbidden keywords). Generates a `constraint_adherence_placeholder` score.
    2.  **Content Analysis (`_analyze_content`)**:
        *   It calls an LLM (e.g., `gpt-4`, configured via its settings) with a specialized prompt. This prompt asks the LLM to evaluate the *received `llm_output`* for quality aspects like relevance, coherence, completeness, accuracy, and safety, based on the original `task_description` and the `prompt_chromosome` (that supposedly generated the `llm_output`).
        *   Parses the LLM's JSON response to extract these scores (e.g., `llm_assessed_relevance`, `llm_assessed_quality`).
        *   Has fallback mechanisms (`_get_fallback_llm_metrics`) if this LLM call fails or returns unparsable data.
    3.  **Fitness Calculation**: Combines the `constraint_adherence_placeholder` score and the `llm_assessed_quality` (from its own LLM analysis of the mock/empty input) using configurable weights (from `fitness_score_weights` in its settings, often loaded from `results_evaluator_config.json`).
*   **Configuration**: Managed through its settings, including LLM provider/model for evaluation, fitness score weights, and path to its knowledge file (e.g., `results_evaluator_config.json`).
*   **Key Issue**: While this agent is sophisticated and uses an LLM, its analysis is performed on the mock or empty output from `FitnessEvaluator`. The meaningfulness of this analysis for the original prompt's quality is therefore limited.

### 2.3. `StyleOptimizerAgent` (e.g., `prompthelix/agents/style_optimizer.py`)

*   **Role**: Intended for "smart" or context-aware mutations of prompts.
*   **Integration**: `GeneticOperators.mutate` can call this agent if it's provided during initialization and a `target_style` is given to the `mutate` function.
*   **Process**: The specific implementation of `StyleOptimizerAgent` would define how it refines a `PromptChromosome` based on a `target_style`. This could involve LLM calls to rephrase, restructure, or add stylistic elements to the prompt's genes.
*   **Status**: The hook for this agent is present in `GeneticOperators`. Its effectiveness depends on its own implementation and how it's utilized in the GA run configuration.

### 2.4. Other Agents

*   **`PromptCriticAgent`**: Potentially used to critique prompts, perhaps by `ResultsEvaluatorAgent` or `PromptArchitectAgent` to refine their outputs or evaluation criteria. Not directly in the main GA loop described in `engine.py`.
*   **`DomainExpertAgent`**: Could provide domain-specific knowledge or evaluation heuristics, possibly consulted by `ResultsEvaluatorAgent`.
*   **`MetaLearnerAgent`**: Designed to consume data from GA runs (e.g., fitness scores, successful prompts) to adapt strategies or configurations over time (e.g., updating agent knowledge files).

## 3. Genetic Algorithm Evaluation Process (Current State)

The evaluation of a `PromptChromosome`'s fitness currently follows these steps:

1.  A GA runner script (e.g., `ga_runner.py`) iterates through generations. For each chromosome in the current generation:
2.  The `PopulationManager`'s `evolve_population` method (or a similar orchestrating function) calls `FitnessEvaluator.evaluate(chromosome, task_description, success_criteria)`.
3.  **Inside `FitnessEvaluator.evaluate()`**:
    a.  The `chromosome`'s genes are converted to a `prompt_string`.
    b.  **No real LLM call is made with this `prompt_string` to get its output.**
    c.  If `execution_mode == ExecutionMode.TEST`, a `mock_llm_output` is generated (e.g., "Mock LLM output for: {prompt_string_snippet}... Random number: {rand_int}").
    d.  Otherwise, `llm_output` is an **empty string**.
    e.  `ResultsEvaluatorAgent.process_request()` is called with the `chromosome`, this `mock_llm_output` (or empty string), `task_description`, and `success_criteria`.
4.  **Inside `ResultsEvaluatorAgent.process_request()`**:
    a.  The agent first checks the `mock_llm_output` against any `success_criteria` (e.g., length constraints, keyword presence). This yields a `constraint_adherence_placeholder` score.
    b.  The agent then calls its **own configured LLM** (e.g., GPT-4). It asks this LLM to evaluate the quality (relevance, coherence, etc.) of the `mock_llm_output` in the context of the original `task_description` and the `prompt_chromosome`'s content.
    c.  The scores from this LLM analysis (e.g., `llm_assessed_quality`) are extracted.
    d.  A final `fitness_score` is calculated by weighting the `constraint_adherence_placeholder` and the `llm_assessed_quality`.
5.  This `fitness_score` is returned to `FitnessEvaluator`, which assigns it to `chromosome.fitness_score`.
6.  After all chromosomes are evaluated, the population is sorted, and the GA proceeds with selection, crossover, and mutation to create the next generation.

## 4. Highlight: What Doesn't Seem to Work (for Real LLM Prompt Optimization)

The primary area where the current GA implementation (as primarily defined by `prompthelix/genetics/engine.py`'s `FitnessEvaluator`) falls short for optimizing prompts for real-world LLM performance is:

*   **Lack of Real LLM Output Generation in the Fitness Loop**:
    *   The `FitnessEvaluator` does not send the candidate `prompt_string` (derived from a chromosome's genes) to the target LLM that the user intends to optimize for. Instead, it generates a mock output or an empty string.
    *   **Implication**: The GA is not evolving prompts based on how they actually perform (i.e., the quality of the output they elicit from a real LLM). It's optimizing prompts based on how well a *mock or empty string* is evaluated by the `ResultsEvaluatorAgent`'s separate LLM analysis.

*   **Misleading Sophistication of `ResultsEvaluatorAgent` in Current Flow**:
    *   While the `ResultsEvaluatorAgent` is sophisticated and uses an LLM to perform detailed analysis, its efforts are spent analyzing a placeholder output.
    *   **Implication**: The fitness scores, while internally consistent with the `ResultsEvaluatorAgent`'s logic, likely do not correlate well with the actual utility or effectiveness of the prompts when used with a real target LLM. The GA might learn to create prompts that lead to "good" mock outputs according to `ResultsEvaluatorAgent`, rather than prompts that are genuinely effective.

*   **`StyleOptimizerAgent`**:
    *   The integration for the `StyleOptimizerAgent` in the mutation process exists.
    *   Its actual impact on producing better prompts depends on (a) the specific implementation of the `StyleOptimizerAgent` itself (how well it refines prompts) and (b) whether it is actively configured and used with appropriate `target_style` inputs during GA runs.

To make the GA effective for optimizing prompts against a specific target LLM, the `FitnessEvaluator` (or a similar component in the evaluation pipeline) would need to be modified to:
1.  Take the `prompt_string` from a chromosome.
2.  Send this `prompt_string` to the actual target LLM.
3.  Capture the real output from that LLM.
4.  Pass this *real LLM output* to the `ResultsEvaluatorAgent` for assessment.

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

The PromptHelix GA provides a flexible framework for prompt evolution with distinct roles for various agents. The `PromptArchitectAgent` uses LLMs effectively to create diverse initial populations. The `ResultsEvaluatorAgent` has a sophisticated mechanism for analyzing text quality using an LLM. However, the crucial link – generating *actual* output from the target LLM using the candidate prompt and evaluating *that* output – is currently missing in the main evaluation flow handled by the `FitnessEvaluator` in `engine.py`. Addressing this would be key to enabling the GA to optimize prompts for real-world LLM performance.