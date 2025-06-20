from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class BaseExperimentRunner(ABC):
    """
    Abstract base class for experiment runners.
    Defines a common interface for running, controlling, and querying experiments.
    """

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the experiment.

        Args:
            **kwargs: Experiment-specific arguments.

        Returns:
            Any: The result of the experiment (e.g., best chromosome, metrics).
        """
        pass

    @abstractmethod
    def pause(self) -> None:
        """
        Pause the currently running experiment, if supported.
        """
        pass

    @abstractmethod
    def resume(self) -> None:
        """
        Resume a paused experiment, if supported.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the currently running experiment.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the experiment.

        Returns:
            Dict[str, Any]: A dictionary containing status information.
                            Common keys might include 'status_message', 'progress',
                            'details', etc.
        """
        pass
