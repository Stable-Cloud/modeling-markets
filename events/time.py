# events/time.py
from events.base import StoppingCondition


class TimeStoppingCondition(StoppingCondition):
    """
    Stop when market time reaches or exceeds a given value.
    """

    def __init__(self, trigger_time):
        self.trigger_time = trigger_time
        self.triggered = False

    def has_triggered(self, market_state):
        if self.triggered:
            return False

        if market_state["time"] >= self.trigger_time:
            self.triggered = True
            return True

        return False
