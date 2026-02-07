# events/price.py
from events.base import StoppingCondition


class PriceStoppingCondition(StoppingCondition):
    """
    Stop when price crosses a specified level.
    """

    def __init__(self, asset_index, level, direction="above"):
        self.asset_index = asset_index
        self.level = level
        self.direction = direction
        self.triggered = False

    def has_triggered(self, market_state):
        if self.triggered:
            return False

        price = market_state["prices"][self.asset_index]

        if self.direction == "above" and price >= self.level:
            self.triggered = True
            return True

        if self.direction == "below" and price <= self.level:
            self.triggered = True
            return True

        return False
