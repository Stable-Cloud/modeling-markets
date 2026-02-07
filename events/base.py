# events/base.py
from abc import ABC, abstractmethod


class StoppingCondition(ABC):
    """
    Abstract base class for stopping-time conditions.
    """

    @abstractmethod
    def has_triggered(self, market_state):
        """
        Return True if the stopping time has occurred
        given the current market state.
        """
        pass
