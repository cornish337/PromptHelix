# Genetic Algorithm Engine in PromptHelix

## Overview

The Genetic Algorithm (GA) is the core engine within PromptHelix responsible for the evolutionary optimization of prompts. Its primary purpose is to iteratively refine a population of prompts (represented as `PromptChromosome` objects) to improve their effectiveness based on defined fitness criteria. This is achieved by simulating natural selection processes, including selection, crossover, and mutation, over multiple generations.

## Core Components

The GA implementation in PromptHelix relies on several key components:

### 1. `PromptChromosome`

-   **Structure**: Each `PromptChromosome` represents a single prompt. It encapsulates:
    -   `id`: A unique identifier (UUID) for the chromosome.
    -   `genes`: A list of strings or structured objects, where each gene represents a segment or component of the prompt (e.g., an instruction, context, specific phrasing).
    -   `fitness_score`: A floating-point number indicating the quality or effectiveness of the prompt, typically assigned by the `FitnessEvaluator`.
-   **Role**: `PromptChromosome` objects are the individuals that make up the population. They are selected, combined (crossover), and modified (mutation) throughout the evolutionary process. Key methods include `to_prompt_string()` for generating the LLM-ready prompt and `clone()` for creating independent copies.

### 2. `GeneticOperators`

This class provides the mechanisms for creating new generations of prompts:

-   **`selection(population, tournament_size)`**:
    -   **Method**: Implements tournament selection.
    -   **Details**: A small subset of the population (the "tournament") is randomly chosen. The individual with the highest `fitness_score` from this subset is selected as a parent for the next generation. This process is typically repeated to select multiple parents.
-   **`crossover(parent1, parent2, crossover_rate)`**:
    -   **Method**: Implements single-point crossover.
    -   **Details**: With a probability defined by `crossover_rate`, two parent chromosomes exchange genetic material. A single crossover point is randomly selected within the (shorter) gene list. Child 1 receives genes from parent 1 up to the crossover point and genes from parent 2 after that point. Child 2 receives the complementary set of genes. If no crossover occurs (based on `crossover_rate`), the children are direct clones of the parents (with new IDs and reset fitness).
-   **`mutate(chromosome, mutation_rate, gene_mutation_prob)`**:
    -   **Method**: Implements gene-level mutation.
    -   **Details**: An individual chromosome has a chance (`mutation_rate`) to undergo mutation. If selected for mutation, each of its genes has a further chance (`gene_mutation_prob`) to be altered.
    -   **Current Mutations**: Implemented mutations are simple string operations (e.g., appending a random character, reversing a slice of the gene string, or replacing the gene with a placeholder like `[MUTATED_GENE_SEGMENT]`).
    -   **Conceptual Agent Call**: The `mutate` method includes a placeholder/comment for a "smart" mutation type (`style_optimization_placeholder`). This conceptually involves calling the `StyleOptimizerAgent` to refine a gene, demonstrating a hook for more intelligent, agent-driven mutations in the future. For now, it appends `[StyleOptimized_Placeholder]` to the gene.

### 3. `FitnessEvaluator`

-   **Role**: Responsible for assessing the quality of each `PromptChromosome`.
-   **Process**:
    1.  It takes a `PromptChromosome` and a `task_description` (and optional `success_criteria`).
    2.  It converts the chromosome's genes into a single `prompt_string`.
    3.  **LLM Interaction Simulation**: It simulates sending this `prompt_string` to an LLM and receiving an output. The current implementation generates a `mock_llm_output` that includes parts of the prompt string and some random elements to mimic variability.
    4.  **Uses `ResultsEvaluatorAgent`**: The `mock_llm_output`, along with the original chromosome, task description, and success criteria, is passed to an instance of `ResultsEvaluatorAgent`. This agent then calculates a `fitness_score` based on its internal (placeholder) logic for metrics like relevance, coherence, and constraint adherence.
    5.  The `fitness_score` obtained from `ResultsEvaluatorAgent` is then assigned back to the `PromptChromosome`.

### 4. `PopulationManager`

-   **Role**: Orchestrates the entire GA lifecycle. It manages the collection of `PromptChromosome` individuals and drives the evolution from one generation to the next.
-   **Initialization**:
    -   The `initialize_population()` method creates the initial set of prompts.
    -   It uses an instance of `PromptArchitectAgent` to generate each `PromptChromosome` based on an initial task description, keywords, and constraints.
-   **Evolutionary Cycle (`evolve_population`)**: This is the main loop that performs the following steps for each generation:
    1.  **Evaluation**: Calculates the fitness of every `PromptChromosome` in the current population using the `FitnessEvaluator`.
    2.  **Sorting**: Sorts the population based on fitness scores in descending order.
    3.  **Elitism**: Carries over a specified number (`elitism_count`) of the fittest individuals directly to the next generation without modification.
    4.  **Reproduction (Offspring Generation)**: Fills the remainder of the new population by:
        *   **Selection**: Selecting parent chromosomes from the current population using `GeneticOperators.selection()`.
        *   **Crossover**: Creating new offspring (children) from selected parents using `GeneticOperators.crossover()`.
        *   **Mutation**: Applying mutations to the offspring using `GeneticOperators.mutate()`.
    5.  The newly generated individuals replace the old population (except for the elites).
    6.  The generation number is incremented.

## GA Flow within `PopulationManager.evolve_population()`

The `evolve_population` method in `PopulationManager` executes the core GA steps in sequence:

1.  **Evaluate Fitness**: For each `PromptChromosome` in the current population, its fitness is determined by the `FitnessEvaluator` (which involves simulating an LLM call and using the `ResultsEvaluatorAgent`).
2.  **Sort Population**: The entire population is sorted by the newly calculated fitness scores, from highest to lowest.
3.  **Apply Elitism**: The top N individuals (defined by `elitism_count`) are directly copied to the next generation's population list.
4.  **Generate Offspring**:
    *   Loop until the new population reaches its target size:
        *   Select two parent chromosomes from the current (sorted) population using the `selection` operator (e.g., tournament selection).
        *   Create two child chromosomes by applying the `crossover` operator to the selected parents.
        *   Apply the `mutation` operator to each child chromosome.
        *   Add the (potentially mutated) children to the new population.
5.  **Replace Old Population**: The `new_population` becomes the current `population` for the next generation.
6.  **Increment Generation Counter**: The `generation_number` is increased.

## Agent Interactions in GA

The GA engine is designed to leverage the specialized capabilities of other agents within the PromptHelix system:

-   **`PromptArchitectAgent`**:
    -   Used by `PopulationManager.initialize_population()` to create the initial diverse set of `PromptChromosome` individuals. The architect designs these initial prompts based on a given task description.
-   **`ResultsEvaluatorAgent`**:
    -   Used by `FitnessEvaluator.evaluate()`. After a `PromptChromosome` is converted to a string and an LLM output is (currently, mock) generated, the `ResultsEvaluatorAgent` assesses this output against the task description and success criteria to produce a fitness score.
-   **`StyleOptimizerAgent` (Conceptual Interaction)**:
    -   The `GeneticOperators.mutate()` method includes a conceptual hook for a "smart" mutation. This placeholder (`style_optimization_placeholder`) indicates where the `StyleOptimizerAgent` could be called to perform more intelligent, context-aware mutations on individual genes or entire prompts, rather than just random character changes. This integration is planned for future development.

Other agents like `PromptCriticAgent` and `DomainExpertAgent` are not directly called within the core GA loop as currently implemented in `engine.py` but are intended to be used by the `ResultsEvaluatorAgent` or potentially by more advanced versions of the `PromptArchitectAgent` or `FitnessEvaluator` to inform their processes. The `MetaLearnerAgent` would consume data produced by the GA run (fitness scores, successful prompts) to adapt strategies over time.Okay, I've created `prompthelix/docs/genetic_algorithm.md` with the detailed explanation of the GA engine.

Now, I will update `prompthelix/docs/README.md` to add the "Genetic Algorithm Engine" section and the instructions for running the orchestrator.
