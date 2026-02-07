# execution/validator.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """
    Result of an execution attempt.
    
    Attributes
    ----------
    success : bool
        Whether the trade was executed
    executed_quantity : float
        Actual quantity executed (may differ from requested)
    price : float
        Execution price
    cost : float
        Total cost (negative for sales)
    reason : str
        Explanation of execution outcome
    """
    success: bool
    executed_quantity: float
    price: float
    cost: float
    reason: str


class ExecutionEngine:
    """
    Validates and executes admissible actions.
    
    Parameters
    ----------
    allow_short_selling : bool, optional
        Whether to allow short positions (default: False)
    allow_fractional : bool, optional
        Whether to allow fractional shares (default: True)
    cost_model : TransactionCostModel, optional
        Model for computing transaction costs (default: None)
    """

    def __init__(self, allow_short_selling=False, allow_fractional=True, cost_model=None):
        self.allow_short_selling = allow_short_selling
        self.allow_fractional = allow_fractional
        self.cost_model = cost_model
    
    def execute(self, agent, action, market_state):
        """
        Execute a trading action for an agent.
        
        Parameters
        ----------
        agent : Agent
            The agent proposing the action
        action : Action
            The proposed trading action
        market_state : dict
            Current market state with prices
        
        Returns
        -------
        ExecutionResult
            Details of the execution outcome
        """
        price = market_state["prices"][action.asset_index]
        quantity = action.quantity
        
        # Handle fractional shares
        if not self.allow_fractional:
            quantity = round(quantity)
            if quantity == 0:
                return ExecutionResult(
                    success=False,
                    executed_quantity=0.0,
                    price=price,
                    cost=0.0,
                    reason="Quantity rounds to zero"
                )
        
        # Check short selling constraint
        if not self.allow_short_selling and quantity < 0:
            current_position = agent.positions.get(action.asset_index, 0.0)
            if current_position + quantity < 0:
                return ExecutionResult(
                    success=False,
                    executed_quantity=0.0,
                    price=price,
                    cost=0.0,
                    reason="Short selling not allowed"
                )
        
        # Compute base cost
        base_cost = price * quantity
        
        # Compute transaction costs
        transaction_cost = 0.0
        if self.cost_model is not None:
            transaction_cost = self.cost_model.compute_cost(quantity, price)
        
        total_cost = base_cost + transaction_cost
        
        # Rule B: self-financing constraint
        if total_cost > agent.capital:
            # Try to execute maximum feasible quantity
            if quantity > 0: #buying case
                max_quantity = agent.capital / (price * (1 + transaction_cost / abs(base_cost)))
                if max_quantity < 0.01:  # Too small to be meaningful
                    return ExecutionResult(
                        success=False,
                        executed_quantity=0.0,
                        price=price,
                        cost=0.0,
                        reason=f"Insufficient capital: need {total_cost:.2f}, have {agent.capital:.2f}"
                    )
            else: #selling case
                return ExecutionResult(
                    success=False,
                    executed_quantity=0.0,
                    price=price,
                    cost=0.0,
                    reason=f"Insufficient capital for transaction costs"
                )
        
        # Execute trade
        agent.capital -= total_cost
        agent.positions[action.asset_index] = (
            agent.positions.get(action.asset_index, 0.0) + quantity
        )
        
        # Track transaction costs in agent
        if hasattr(agent, 'total_transaction_costs'):
            agent.total_transaction_costs += transaction_cost
        
        # Track trade count
        if hasattr(agent, 'trade_count'):
            agent.trade_count += 1
        
        return ExecutionResult(
            success=True,
            executed_quantity=quantity,
            price=price,
            cost=total_cost,
            reason="Trade executed successfully"
        )

