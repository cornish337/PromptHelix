import asyncio

from prompthelix.message_bus import MessageBus
import logging
import time  # Added
import asyncio # Added import

logger = logging.getLogger(__name__)

# Set a default for SessionLocal so references outside main_ga_loop do not fail
SessionLocal = None
from prompthelix.globals import websocket_manager  # Use the global connection manager
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.agents.domain_expert import DomainExpertAgent # Added for demo
from prompthelix.agents.critic import PromptCriticAgent # Added for demo
# Import BaseAgent if we need to use it for the example, or a mock agent
from prompthelix.agents.base import BaseAgent
from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner # Changed import
from prompthelix.genetics.engine import (
    GeneticOperators,
    FitnessEvaluator,
    PopulationManager,
    PromptChromosome,
    # FitnessEvaluator, # No longer directly imported if loaded by path
)
import importlib # Added for dynamic class loading
from typing import List, Optional, Dict, Type # Added Type
from prompthelix.enums import ExecutionMode
from prompthelix.utils.config_utils import update_settings # Assuming a utility for deep merging configs
from prompthelix import config as global_config # To access global default settings
from prompthelix.config import settings, ENABLE_WANDB_LOGGING, WANDB_PROJECT_NAME, WANDB_ENTITY_NAME # Added import
# "old"

from prompthelix import config as global_ph_config # Renamed to avoid conflict with local 'config' variable
from prompthelix.config import settings as global_settings_obj, AGENT_PIPELINE_CONFIG # Import AGENT_PIPELINE_CONFIG directly
from prompthelix.genetics.fitness_base import BaseFitnessEvaluator # For type hinting

from prompthelix.utils import start_exporter_if_enabled, update_generation, update_best_fitness
from prompthelix import config as global_ph_config  # renamed to avoid clash with local `config`
from prompthelix.config import settings as global_settings_obj  # for mutation / selection / crossover strategy classes
from prompthelix.config import settings  # for WANDB / MLflow keys, etc.
from prompthelix.genetics.fitness_base import BaseFitnessEvaluator  # fitness-evaluator ABC

try:
    import wandb
except Exception as e:  # broaden exception handling to avoid startup failure
    logger.warning(
        "wandb failed to import. W&B logging will be disabled. Error: %s", e
    )
    wandb = None


async def main_ga_loop( # Changed to async def
    task_desc: str,
    keywords: List[str],
    num_generations: int,
    population_size: int,
    elitism_count: int,
    execution_mode: ExecutionMode,
    initial_prompt_str: Optional[str] = None,
    agent_settings_override: Optional[Dict] = None,
    llm_settings_override: Optional[Dict] = None,
    parallel_workers: Optional[int] = None,
    return_best: bool = True,
    population_path: Optional[str] = None,
    save_frequency_override: Optional[int] = None,
    metrics_file_path: Optional[str] = None,
):
    """
    Main orchestration loop for running the PromptHelix Genetic Algorithm.

    Args:
        task_desc: Description of the task prompts should solve.
        keywords: Keywords used by the PromptArchitectAgent.
        num_generations: Number of generations to evolve.
        population_size: Desired population size.
        elitism_count: Number of top individuals preserved each generation.
        execution_mode: Whether to run in REAL or TEST mode.
        initial_prompt_str: Optional initial prompt string to seed population.
        agent_settings_override: Optional dictionary to override agent settings.
        llm_settings_override: Optional dictionary to override LLM utility settings.
        parallel_workers: Number of parallel workers for fitness evaluation.
        return_best: If True, return the best chromosome at the end.
        population_path: Optional path for loading/saving population. If None, uses config default.
        save_frequency_override: Optional override for population save frequency. If None, uses config default.
        metrics_file_path: Optional path to write generation metrics as JSON lines.
    """
    # Import SessionLocal lazily so that SQLAlchemy is only required when
    # database features are used.
    try:
        from prompthelix.database import SessionLocal as _SessionLocal  # type: ignore
    except ModuleNotFoundError:
        logger.error(
            "SQLAlchemy is required for database logging. Install it with 'pip install SQLAlchemy'."
        )
        _SessionLocal = None

    # Local reference used throughout this function and update module-level
    global SessionLocal
    SessionLocal = _SessionLocal

    logger.info("--- main_ga_loop started ---")
    logger.info(f"Task Description: {task_desc}")
    logger.info(f"Keywords: {keywords}")
    logger.info(f"Num Generations: {num_generations}, Population Size: {population_size}, Elitism Count: {elitism_count}")
    logger.info(f"Execution Mode: {execution_mode.name}")
    start_exporter_if_enabled()
    if initial_prompt_str:
        logger.info(f"Initial Prompt String provided: '{initial_prompt_str[:100]}...'")
    if agent_settings_override:
        logger.info(f"Agent Settings Override provided: {agent_settings_override}")
    if llm_settings_override:
        logger.info(f"LLM Settings Override provided: {llm_settings_override}")
    if parallel_workers is not None:
        logger.info(f"Parallel Workers specified: {parallel_workers}")
    else:
        logger.info("Parallel Workers: Using default (None).")
    if population_path is not None:
        logger.info(f"Population path override provided: {population_path}")
    if save_frequency_override is not None:
        logger.info(f"Save frequency override provided: {save_frequency_override}")

    # Determine actual persistence settings
    actual_population_path = population_path if population_path is not None else settings.DEFAULT_POPULATION_PERSISTENCE_PATH
    actual_save_frequency = save_frequency_override if save_frequency_override is not None else settings.DEFAULT_SAVE_POPULATION_FREQUENCY

    logger.info(f"Effective Population Persistence Path: {actual_population_path}")
    logger.info(f"Effective Save Population Frequency: Every {actual_save_frequency} generations (0 means periodic saving disabled)")
    if metrics_file_path:
        logger.info(f"Generation metrics will be written to: {metrics_file_path}")

    current_wandb_enabled = ENABLE_WANDB_LOGGING and wandb is not None
    wandb_run = None

    if current_wandb_enabled:
        try:
            wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_ENTITY_NAME, # Optional: Can be None
                config={
                    "task_description": task_desc,
                    "keywords": ",".join(keywords) if keywords else "",
                    "num_generations": num_generations,
                    "population_size": population_size,
                    "elitism_count": elitism_count,
                    "execution_mode": execution_mode.name,
                    "initial_prompt_provided": bool(initial_prompt_str),
                    "parallel_workers": parallel_workers,
                    "effective_population_path": actual_population_path,
                    "effective_save_frequency": actual_save_frequency,
                    "agent_settings_override": agent_settings_override if agent_settings_override else "None",
                    "llm_settings_override": llm_settings_override if llm_settings_override else "None",
                }
            )
            logger.info(f"W&B run initialized: {wandb_run.name if wandb_run else 'Failed'}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}", exc_info=True)
            current_wandb_enabled = False # Disable W&B if init fails

    # 0. Instantiate Message Bus
    logger.debug("Initializing Message Bus...")
    if SessionLocal is not None:
        message_bus = MessageBus(
            db_session_factory=SessionLocal, connection_manager=websocket_manager
        )
    else:
        logger.warning(
            "SessionLocal is not available. MessageBus will run without database logging."
        )
        message_bus = MessageBus(connection_manager=websocket_manager)
    logger.debug("Message Bus initialized.")

    # Handle LLM settings override (used by default FitnessEvaluator and potentially others)
    current_llm_settings = update_settings(global_ph_config.LLM_UTILS_SETTINGS.copy(), llm_settings_override)
    if llm_settings_override:
        logger.info("LLM settings have been updated with overrides for this session.")


    logger.info("Initializing agents from AGENT_PIPELINE_CONFIG...")
    loaded_agents: Dict[str, BaseAgent] = {}
    agent_names: List[str] = []

    for agent_conf in AGENT_PIPELINE_CONFIG: # Use the directly imported AGENT_PIPELINE_CONFIG
        class_path = agent_conf.get("class_path")
        agent_id = agent_conf.get("id")
        settings_key = agent_conf.get("settings_key")

        if not all([class_path, agent_id, settings_key]):
            logger.error(f"Invalid agent configuration: {agent_conf}. Skipping.")
            continue

        logger.info(f"Loading agent '{agent_id}' from class path '{class_path}' using settings key '{settings_key}'.")
        try:
            module_path_str, class_name_str = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path_str)
            AgentClass: Type[BaseAgent] = getattr(module, class_name_str)

            # Get base settings from global AGENT_SETTINGS using settings_key
            agent_base_settings = global_ph_config.AGENT_SETTINGS.get(settings_key, {}).copy()
            # Apply overrides passed to main_ga_loop (if any for this agent_id or settings_key)
            # Note: agent_settings_override might be keyed by agent_id or settings_key. Assuming settings_key for now.
            specific_override = {}
            if agent_settings_override:
                specific_override = agent_settings_override.get(settings_key, agent_settings_override.get(agent_id, {}))

            final_agent_settings = update_settings(agent_base_settings, specific_override)

            # Instantiate agent
            # BaseAgent constructor: __init__(self, agent_id: str, message_bus=None, settings: Optional[Dict] = None)
            # Specific agents might have more params (e.g. knowledge_file_path directly)
            # For now, assume agents can get all they need from their 'settings' dict or have defaults.
            # If knowledge_file_path is standard, it should be in final_agent_settings.
            # The individual agent classes need to be robust to this.
            agent_instance = AgentClass(
                agent_id=agent_id, # Use the ID from pipeline config
                message_bus=message_bus,
                settings=final_agent_settings
            )

            loaded_agents[agent_id] = agent_instance
            message_bus.register(agent_id, agent_instance)
            agent_names.append(agent_id)
            logger.info(f"Successfully loaded and registered agent: {agent_id} (Class: {AgentClass.__name__})")

        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Failed to load or instantiate agent '{agent_id}' from '{class_path}': {e}", exc_info=True)
            # Decide if this is a critical failure or if the GA can proceed without this agent

    if not loaded_agents:
        logger.error("No agents were loaded from the pipeline configuration. GA cannot proceed effectively.")
        # Depending on requirements, either raise an error or try to continue with minimal functionality (if possible)
        raise ValueError("Agent pipeline configuration resulted in no loaded agents.")

    logger.info(f"All configured agents loaded and registered. Agent IDs: {agent_names}")

    # Retrieve specific agents needed by core GA logic by their configured ID.
    # These IDs must be present in the AGENT_PIPELINE_CONFIG.
    # TODO: Make these required agent IDs configurable or handle their absence more gracefully.
    prompt_architect = loaded_agents.get("PromptArchitectAgent")
    results_evaluator = loaded_agents.get("ResultsEvaluatorAgent") # Needed for default FitnessEvaluator
    style_optimizer = loaded_agents.get("StyleOptimizerAgent") # Passed to GeneticOperators

    if not prompt_architect:
        raise ValueError("Required 'PromptArchitectAgent' not found in loaded agents.")
    if not results_evaluator: # Critical for the default fitness evaluator
        raise ValueError("Required 'ResultsEvaluatorAgent' not found in loaded agents (needed for default fitness evaluation).")


    # 2. Instantiate GA Components
    logger.debug("Initializing GA components...")

    # Pass style_optimizer_config to GeneticOperators if it needs settings
    metrics_logger_instance = logging.getLogger("prompthelix.ga_metrics")
    genetic_ops = GeneticOperators(
        style_optimizer_agent=style_optimizer,
        # metrics_logger=metrics_logger_instance # metrics_logger is not an expected argument
        mutation_strategies=None, # Pass None to use default, or load from config
        strategy_settings=global_ph_config.AGENT_SETTINGS.get("GeneticOperatorsStrategySettings")
    )
# old
    # Pass the potentially None style_optimizer to GeneticOperators
    # genetic_ops = GeneticOperators(style_optimizer_agent=style_optimizer, strategy_settings=global_ph_config.AGENT_SETTINGS.get("GeneticOperatorsStrategySettings"))


    # Load and instantiate FitnessEvaluator from configuration
    fitness_evaluator_class_path = global_settings_obj.FITNESS_EVALUATOR_CLASS
    logger.info(f"Loading FitnessEvaluator from: {fitness_evaluator_class_path}")
    try:
        module_path, class_name = fitness_evaluator_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        FitnessEvaluatorClass: Type[BaseFitnessEvaluator] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load FitnessEvaluator class '{fitness_evaluator_class_path}': {e}. Falling back to default.", exc_info=True)
        # Fallback to the default FitnessEvaluator directly if loading fails
        from prompthelix.genetics.engine import FitnessEvaluator as DefaultFitnessEvaluator
        FitnessEvaluatorClass = DefaultFitnessEvaluator

    # Prepare settings/arguments for the chosen FitnessEvaluator
    # The default FitnessEvaluator expects results_evaluator_agent, execution_mode, llm_settings.
    # Custom evaluators might take these via a 'settings' dict or specific kwargs.
    evaluator_init_kwargs = {
        "results_evaluator_agent": results_evaluator, # Specific to default evaluator
        "execution_mode": execution_mode,             # Specific to default evaluator
        "llm_settings": current_llm_settings,         # Specific to default evaluator
        "settings": { # General settings for any evaluator
            "llm_settings": current_llm_settings, # Can be accessed via settings.get('llm_settings')
            "execution_mode": execution_mode.name, # Pass mode as string in general settings
            # Add other general config if needed by custom evaluators
        }
    }

    try:
        fitness_eval_instance = FitnessEvaluatorClass(**evaluator_init_kwargs)
        logger.info(f"Successfully instantiated FitnessEvaluator: {FitnessEvaluatorClass.__name__}")
    except TypeError as te:
        logger.error(f"TypeError instantiating {FitnessEvaluatorClass.__name__} with provided arguments: {te}. Check constructor signature.", exc_info=True)
        # Potentially re-raise or use a very basic fallback if critical
        raise ValueError(f"Could not instantiate FitnessEvaluator {FitnessEvaluatorClass.__name__}") from te
#

    fitness_eval = fitness_eval_instance

    pop_manager = PopulationManager(
        genetic_operators=genetic_ops,
        fitness_evaluator=fitness_eval,
        prompt_architect_agent=prompt_architect,  # Architect is used for initial prompt generation
        population_size=population_size,
        elitism_count=elitism_count,
        initial_prompt_str=initial_prompt_str,
        parallel_workers=parallel_workers,
        message_bus=message_bus,  # Added
        population_path=actual_population_path,

# old
#        metrics_file_path=metrics_file_path,

#

        # TODO: Pass agent_settings_override or specific agent configs if PopulationManager
        # is responsible for creating/configuring more agents during its operations.
        # For now, agents are configured above.
    )
    # PopulationManager accepts `initial_prompt_str` and uses it when initializing
    # the population.
    logger.debug("GA components initialized.")

    # 3. Use GA Parameters (already logged)
    evaluation_task_desc = task_desc
    evaluation_success_criteria = {
        "must_include_keywords": keywords[:2] if keywords else [],
        "max_length": 500 # This could also come from config or overrides
    }
    logger.info(f"Evaluation criteria (example): {evaluation_success_criteria}")


    # 4. Initialize Population
    if not pop_manager.population:
        logger.info("--- Initializing Population ---")
        # PopulationManager uses initial_prompt_str if provided
        await pop_manager.initialize_population( # Added await
            initial_task_description=task_desc, # This might be redundant if initial_prompt_str is primary
            initial_keywords=keywords
        )
        logger.info(f"Population initialized with {len(pop_manager.population)} individuals.")
    else:
        print(f"Loaded population with {len(pop_manager.population)} individuals.")

    # Evaluate the initial population to see a starting best
    # (evolve_population also does this, but this shows the state *before* any evolution)
    if pop_manager.population:
        print("Evaluating initial population...")
        # This loop needs to be async if evaluate is async
        initial_evaluation_tasks = []
        for chromo in pop_manager.population:
            # Ensure fitness is evaluated based on the primary task and criteria
            initial_evaluation_tasks.append(
                fitness_eval.evaluate(chromo, evaluation_task_desc, evaluation_success_criteria)
            )
        if initial_evaluation_tasks:
            await asyncio.gather(*initial_evaluation_tasks) # Await all evaluations

        pop_manager.population.sort(key=lambda c: c.fitness_score, reverse=True)
        fittest_initial = pop_manager.get_fittest_individual()
        if fittest_initial:
            print("\n--- Best in Initial Population ---")
            print(f"Fitness: {fittest_initial.fitness_score:.4f}")
            print(f"Prompt ID: {fittest_initial.id}")
            # print(f"Prompt Genes: {fittest_initial.genes}") # Genes can be verbose
            print(f"Prompt String Snippet: {fittest_initial.to_prompt_string()[:200]}...")
        else:
            print("No individuals in the initial population after evaluation (this shouldn't happen if initialized).")
    else:
        print("Population is empty after initialization. Cannot proceed.")
        if return_best:
            return None # Return None if population is empty
        else:
            return

    # 5. Instantiate and Run GeneticAlgorithmRunner
    logger.info("\n--- Initializing and Starting Genetic Algorithm Runner ---")

    runner = GeneticAlgorithmRunner(
        population_manager=pop_manager,
        num_generations=num_generations,
        save_frequency=actual_save_frequency, # Pass determined save frequency
        population_persistence_path=actual_population_path # Pass the path
    )

    # evaluation_task_desc and evaluation_success_criteria are defined earlier in main_ga_loop
    # For example:
    # evaluation_task_desc = task_desc
    # evaluation_success_criteria = {
    #     "must_include_keywords": keywords[:2] if keywords else [],
    #     "max_length": 500
    # }

    run_kwargs = {
        "task_description": evaluation_task_desc,
        "success_criteria": evaluation_success_criteria,
        "target_style": "formal" # Maintaining the style from the original evolve_population call
    }

    logger.info(f"Handing over to GeneticAlgorithmRunner (ID: {id(runner)}) to run for {num_generations} generations.")
    # The runner's run() method will now manage the evolution loop and pause/stop/complete states.
    fittest_individual = await runner.run(**run_kwargs) # Added await

    logger.info(f"GeneticAlgorithmRunner has completed. Orchestrator check on PopulationManager status: {pop_manager.status}")

    # After the runner completes (or is stopped), get the final best from the population manager
    final_fittest_overall = pop_manager.get_fittest_individual()

    # Consolidate final status update logic (this was present in original orchestrator)
    # This might need adjustment if the runner already robustly sets these.
    # For now, keeping it to ensure final status is logged by orchestrator.
    if pop_manager.status not in ["STOPPED", "COMPLETED", "ERROR"]:
        logger.warning(f"Orchestrator: PopulationManager status is '{pop_manager.status}' after runner completion. Expected STOPPED, COMPLETED, or ERROR.")
        if pop_manager.should_stop: # if a stop was signaled to PM but runner didn't set it
            pop_manager.status = "STOPPED"
        elif pop_manager.generation_number >= num_generations: # Check actual generations run vs target
             pop_manager.status = "COMPLETED"
        else: # If neither stopped nor completed full generations (e.g. error or unexpected exit)
            pop_manager.status = "UNKNOWN_POST_RUN"
        logger.info(f"Orchestrator: Adjusted PopulationManager status to '{pop_manager.status}'.")
        pop_manager.broadcast_ga_update(event_type="ga_run_status_adjusted_orchestrator")


    if final_fittest_overall:
        print("\n--- Overall Best Prompt Found (via Orchestrator) ---")
        print(str(final_fittest_overall))
        logger.info(
            "Orchestrator: GA finished - final population size: %d, best fitness: %.4f, status: %s",
            len(pop_manager.population) if pop_manager.population else 0,
            final_fittest_overall.fitness_score,
            pop_manager.status
        )
    else:
        # This might occur if fittest_individual from runner.run() was None,
        # and pop_manager.get_fittest_individual() also returns None.
        print("\nNo solution found after all generations (checked by Orchestrator).")
        logger.info(
            "Orchestrator: GA finished - final population size: %d, no valid solution, status: %s",
            len(pop_manager.population) if pop_manager.population else 0,
            pop_manager.status
        )

    # Save the final population if a path is provided (regardless of periodic saving)
    # This ensures the final state is always saved if a path is specified.
    if actual_population_path:
        logger.info(f"Orchestrator: Attempting to save final population to {actual_population_path}")
        try:
            pop_manager.save_population()
            logger.info(f"Orchestrator: Final population successfully saved to {actual_population_path}.")
        except Exception as e:
            logger.error(f"Orchestrator: Failed to save final population to {actual_population_path}. Error: {e}", exc_info=True)


    if wandb_run:
        # Optionally log summary metrics to W&B
        if final_fittest_overall:
            wandb_run.summary["final_best_fitness"] = final_fittest_overall.fitness_score
            wandb_run.summary["final_best_prompt"] = final_fittest_overall.to_prompt_string()
        wandb_run.summary["final_status"] = pop_manager.status
        wandb_run.finish()
        logger.info("W&B run finished.")

    if return_best:
        # Ensure GA_RUNNING_STATUS is set to 0 as the run is complete
        from prompthelix import metrics as ph_metrics
        ph_metrics.GA_RUNNING_STATUS.set(0)
        logger.info("Orchestrator: main_ga_loop completed. Set GA_RUNNING_STATUS to 0.")
        return final_fittest_overall
    else: # Ensure it's set even if not returning best
        from prompthelix import metrics as ph_metrics
        ph_metrics.GA_RUNNING_STATUS.set(0)
        logger.info("Orchestrator: main_ga_loop completed. Set GA_RUNNING_STATUS to 0.")


