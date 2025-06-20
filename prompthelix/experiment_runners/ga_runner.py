from typing import Optional, Dict, Any
import logging

# Add these imports at the top of prompthelix/experiment_runners/ga_runner.py
from prompthelix.logging_handlers import WebSocketLogHandler
from prompthelix.globals import websocket_manager # Import the global instance from globals

from prompthelix.experiment_runners.base_runner import BaseExperimentRunner
from prompthelix.genetics.engine import PopulationManager, PromptChromosome

logger = logging.getLogger(__name__)

# Add the WebSocketLogHandler to the logger instance
# This setup can be done once when the module is loaded,
# or specifically when a GeneticAlgorithmRunner is instantiated if preferred.
# For simplicity, let's add it to the module-level logger.
# Ensure this part of the code runs when the module is imported.

# Create and configure the WebSocket log handler
websocket_log_handler = WebSocketLogHandler(connection_manager=websocket_manager)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
websocket_log_handler.setFormatter(formatter)
websocket_log_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity

# Add the handler to the logger
if not any(isinstance(h, WebSocketLogHandler) for h in logger.handlers):
    logger.addHandler(websocket_log_handler)
    logger.info("WebSocketLogHandler added to ga_runner logger.")
else:
    logger.info("WebSocketLogHandler already present in ga_runner logger.")


class GeneticAlgorithmRunner(BaseExperimentRunner):
    """
    Runs a genetic algorithm experiment for a specified number of generations.
    """

    def __init__(self, population_manager: PopulationManager, num_generations: int):
        """
        Initializes the GeneticAlgorithmRunner.

        Args:
            population_manager: A pre-configured PopulationManager instance.
            num_generations: The total number of generations to run.
        """
        if not isinstance(population_manager, PopulationManager):
            raise TypeError("population_manager must be an instance of PopulationManager.")
        if not isinstance(num_generations, int) or num_generations <= 0:
            raise ValueError("num_generations must be a positive integer.")

        self.population_manager = population_manager
        self.num_generations = num_generations
        self.current_generation = 0 # Track current generation internally
        logger.info(
            f"GeneticAlgorithmRunner initialized for {num_generations} generations "
            f"with PopulationManager (ID: {id(population_manager)})"
        )

    def run(self, **kwargs) -> Optional[PromptChromosome]:
        """
        Executes the genetic algorithm for the specified number of generations.

        The 'kwargs' are not directly used by this GA runner's core loop but are
        included for interface compatibility. Specific parameters for GA evolution
        (like task_description, success_criteria, target_style) should be passed
        to the PopulationManager's evolve_population method, typically managed
        by the entity that sets up and calls this runner.

        For this implementation, we'll assume that `task_description`,
        `success_criteria`, and `target_style` are passed via kwargs to `run`
        and then forwarded to `evolve_population`. This makes the runner more
        flexible if these parameters change per run invocation.

        Args:
            **kwargs: Expected to contain:
                - task_description (str): Description of the task for evaluation.
                - success_criteria (Optional[Dict]): Criteria for successful evaluation.
                - target_style (Optional[str]): Target style for style optimization.

        Returns:
            Optional[PromptChromosome]: The best chromosome found after all generations
                                        or if the run was stopped.
        """
        logger.info(f"GeneticAlgorithmRunner: Starting run for {self.num_generations} generations.")
        self.population_manager.status = "RUNNING" # Ensure status is RUNNING at start
        self.population_manager.broadcast_ga_update(event_type="ga_run_started_runner")

        # Extract GA evolution parameters from kwargs
        task_description = kwargs.get("task_description")
        if task_description is None:
            logger.warning("GeneticAlgorithmRunner.run: 'task_description' not provided in kwargs. Fitness evaluation might be impaired.")
        success_criteria = kwargs.get("success_criteria")
        target_style = kwargs.get("target_style")


        fittest_individual: Optional[PromptChromosome] = None

        try:
            for i in range(self.num_generations):
                self.current_generation = self.population_manager.generation_number + 1 # or i + 1
                logger.info(f"GeneticAlgorithmRunner: Starting Generation {self.current_generation}/{self.num_generations}")

                if self.population_manager.should_stop:
                    logger.info(f"GeneticAlgorithmRunner: Stop signal received before Generation {self.current_generation}. Stopping run.")
                    self.population_manager.status = "STOPPED"
                    self.population_manager.broadcast_ga_update(event_type="ga_run_stopped_runner_signal")
                    break

                # evolve_population itself checks for pause/stop and updates status
                self.population_manager.evolve_population(
                    task_description=task_description,
                    success_criteria=success_criteria,
                    target_style=target_style
                )

                # If evolve_population itself detected a stop, it would set status to STOPPED
                if self.population_manager.status == "STOPPED":
                    logger.info(f"GeneticAlgorithmRunner: PopulationManager entered STOPPED state during Generation {self.current_generation}. Stopping run.")
                    # population_manager should have broadcasted its stop
                    break

                fittest_in_gen = self.population_manager.get_fittest_individual()
                if fittest_in_gen:
                    logger.info(
                        f"GeneticAlgorithmRunner: Fittest in Generation {self.population_manager.generation_number}: "
                        f"Fitness={fittest_in_gen.fitness_score:.4f}, ID={fittest_in_gen.id}"
                    )
                    fittest_individual = fittest_in_gen
                else:
                    logger.warning(f"GeneticAlgorithmRunner: No fittest individual found in Generation {self.population_manager.generation_number}.")

                # Check for stop signal again after evolution, in case it was set during the generation
                if self.population_manager.should_stop and self.population_manager.status != "STOPPED":
                    logger.info(f"GeneticAlgorithmRunner: Stop signal detected after Generation {self.current_generation}. Loop will terminate.")
                    self.population_manager.status = "STOPPED" # Ensure status is updated if not already
                    self.population_manager.broadcast_ga_update(event_type="ga_run_stopped_runner_post_gen")


            # After the loop
            if self.population_manager.status not in ["STOPPED", "COMPLETED", "ERROR"]:
                if self.current_generation >= self.num_generations and not self.population_manager.should_stop:
                    logger.info(f"GeneticAlgorithmRunner: Completed all {self.num_generations} generations.")
                    self.population_manager.status = "COMPLETED"
                    self.population_manager.broadcast_ga_update(event_type="ga_run_completed_runner")
                elif self.population_manager.should_stop: # Should have been caught by loop breaks
                    logger.info("GeneticAlgorithmRunner: Run ended due to stop signal (final check).")
                    if self.population_manager.status != "STOPPED": # Defensive
                         self.population_manager.status = "STOPPED"
                         self.population_manager.broadcast_ga_update(event_type="ga_run_stopped_runner_final_check")
                else:
                    # Unclear state, perhaps num_generations was 0 or loop exited unexpectedly
                    logger.warning(f"GeneticAlgorithmRunner: Loop finished, but status is '{self.population_manager.status}'. Generations run: {self.current_generation}/{self.num_generations}.")
                    # Consider setting to COMPLETED or ERROR based on context
                    self.population_manager.status = "UNKNOWN_COMPLETION"


        except Exception as e:
            logger.error(f"GeneticAlgorithmRunner: An error occurred during the run: {e}", exc_info=True)
            self.population_manager.status = "ERROR"
            self.population_manager.broadcast_ga_update(event_type="ga_run_error_runner", additional_data={"error": str(e)})
            # Re-raise the exception so the caller (orchestrator) is aware
            raise
        finally:
            final_fittest = self.population_manager.get_fittest_individual()
            logger.info(
                f"GeneticAlgorithmRunner: Run finished. Status: {self.population_manager.status}. "
                f"Final best fitness: {final_fittest.fitness_score if final_fittest else 'N/A'}."
            )
            # The orchestrator might want to save the population, or this runner could.
            # For now, let's assume the orchestrator handles saving if population_path was provided to PopulationManager.

        return self.population_manager.get_fittest_individual() # Return the best one found

    def pause(self) -> None:
        """Pauses the GA evolution by calling the PopulationManager's method."""
        logger.info("GeneticAlgorithmRunner: Received pause request.")
        self.population_manager.pause_evolution()

    def resume(self) -> None:
        """Resumes the GA evolution by calling the PopulationManager's method."""
        logger.info("GeneticAlgorithmRunner: Received resume request.")
        self.population_manager.resume_evolution()

    def stop(self) -> None:
        """Stops the GA evolution by calling the PopulationManager's method."""
        logger.info("GeneticAlgorithmRunner: Received stop request.")
        self.population_manager.stop_evolution()

    def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the GA from the PopulationManager.
        Also includes the runner's own current generation and target generations.
        """
        pm_status = self.population_manager.get_ga_status()
        runner_status = {
            "runner_current_generation": self.current_generation,
            "runner_target_generations": self.num_generations,
            "runner_population_manager_id": id(self.population_manager)
        }
        # Merge PM status with runner status, giving PM status precedence for shared keys like 'status'
        # although get_ga_status() returns a dict with keys like 'status', 'generation', etc.
        # which are distinct from the runner's specific keys here.
        # A simple update should be fine.
        status_report = pm_status.copy()
        status_report.update(runner_status)
        return status_report
