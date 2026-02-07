# agents/simple.py

from agents.base import Agent
from agents.actions import Action


class SimpleReactiveAgent(Agent):
    """
    A minimal agent that reacts to information cards
    by attempting to buy a fixed quantity of an asset.
    """

    def __init__(self, capital, asset_index=0, trade_size=1.0):
        super().__init__(capital)
        self.asset_index = asset_index
        self.trade_size = trade_size

    def observe_and_propose(self, card, market_state):
        """
        Ignore card content for now.
        Always attempt to buy a fixed quantity.
        """
        return Action(
            asset_index=self.asset_index,
            quantity=self.trade_size
        )
