# agents/base.py
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for agents.
    
    Attributes
    ----------
    capital : float
        Available cash for trading
    positions : dict
        Dictionary mapping asset_index to quantity held
    trade_count : int
        Number of trades executed
    total_transaction_costs : float
        Cumulative transaction costs paid
    """

    def __init__(self, capital):
        self.capital = float(capital)
        self.positions = {}
        self.trade_count = 0
        self.total_transaction_costs = 0.0

    @abstractmethod
    def observe_and_propose(self, card, market_state):
        """
        Observe an information card and market state,
        and propose an action.
        """
        pass

    def update_after_execution(self, executed_action):
        """
        Update internal state after an action is executed.
        
        Parameters
        ----------
        executed_action : ExecutionResult
            Result of the execution attempt
        """
        pass
    
    def get_portfolio_summary(self):
        """
        Get summary of agent's portfolio state.
        
        Returns
        -------
        dict
            Dictionary with capital, positions, trade_count, and costs
        """
        return {
            'capital': self.capital,
            'positions': dict(self.positions),
            'trade_count': self.trade_count,
            'total_transaction_costs': self.total_transaction_costs
        }
