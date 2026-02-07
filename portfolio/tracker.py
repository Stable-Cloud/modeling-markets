# portfolio/tracker.py

class PortfolioTracker:
    """
    Tracks portfolio value and P&L for an agent.
    """

    def __init__(self, agent):
        self.agent = agent
        self.history = []

    def record(self, market_state):
        """
        Record portfolio value at the current market state.
        """
        value = self.compute_value(market_state)
        self.history.append({
            "time": market_state["time"],
            "value": value,
            "cash": self.agent.capital,
            "positions": dict(self.agent.positions)
        })

    def compute_value(self, market_state):
        value = self.agent.capital
        prices = market_state["prices"]

        for asset_index, quantity in self.agent.positions.items():
            value += quantity * prices[asset_index]

        return value
