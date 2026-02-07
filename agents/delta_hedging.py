# agents/delta_hedging.py

from agents.base import Agent
from agents.actions import Action
from options.black_scholes import black_scholes_delta, black_scholes_gamma


class DeltaHedgingAgent(Agent):
    """
    Agent that implements delta hedging strategy for a European call option.
    
    At each rebalancing time, the agent computes the Black-Scholes delta
    and adjusts its position to match the target delta.
    
    Parameters
    ----------
    capital : float
        Initial cash available
    option_params : dict
        Dictionary with option parameters: {'K', 'r', 'sigma', 'T'}
    asset_index : int, optional
        Index of the asset to hedge (default: 0)
    rebalance_tolerance : float, optional
        Minimum delta deviation to trigger rebalancing (default: 0.001)
        Avoids excessive trading for tiny adjustments
    """
    
    def __init__(self, capital, option_params, asset_index=0, rebalance_tolerance=0.001):
        super().__init__(capital)
        self.option_params = option_params
        self.asset_index = asset_index
        self.rebalance_tolerance = rebalance_tolerance
        
        # Validate option parameters
        required_keys = {'K', 'r', 'sigma', 'T'}
        if not required_keys.issubset(option_params.keys()):
            raise ValueError(
                f"option_params must contain {required_keys}. "
                f"Got: {set(option_params.keys())}"
            )
    
    def compute_target_delta(self, S, t):
        """
        Compute the target delta position using Black-Scholes formula.
        
        Parameters
        ----------
        S : float
            Current stock price
        t : float
            Current time
        
        Returns
        -------
        float
            Target delta (number of shares to hold)
        """
        K = self.option_params['K']
        r = self.option_params['r']
        sigma = self.option_params['sigma']
        T = self.option_params['T']
        
        # Handle edge case: at or past maturity
        if t >= T:
            return 1.0 if S > K else 0.0
        
        return black_scholes_delta(S, K, r, sigma, T, t)
    
    def observe_and_propose(self, card, market_state):
        """
        Observe market state and propose rebalancing trade.
        
        Computes target delta and proposes trade to adjust position.
        Uses tolerance to avoid excessive trading.
        
        Parameters
        ----------
        card : InformationCard
            Information revealed at this stopping time
        market_state : dict
            Current market state with 'time' and 'prices'
        
        Returns
        -------
        Action
            Proposed rebalancing trade
        """
        S = market_state["prices"][self.asset_index]
        t = market_state["time"]
        
        # Compute target delta
        target_delta = self.compute_target_delta(S, t)
        
        # Get current position
        current_position = self.positions.get(self.asset_index, 0.0)
        
        # Compute required trade
        required_trade = target_delta - current_position
        
        # Apply tolerance: don't trade if adjustment is tiny
        if abs(required_trade) < self.rebalance_tolerance:
            required_trade = 0.0
        
        return Action(
            asset_index=self.asset_index,
            quantity=required_trade
        )
    
    def update_after_execution(self, execution_result):
        """
        Update internal state after execution.
        
        Parameters
        ----------
        execution_result : ExecutionResult
            Result of the execution attempt
        """
        # Base class already tracks trade_count and costs via ExecutionEngine
        pass


class AdaptiveDeltaHedgingAgent(DeltaHedgingAgent):
    """
    Delta hedging agent with adaptive rebalancing tolerance.
    
    Adjusts rebalancing threshold based on gamma: higher gamma
    (more convexity) leads to tighter tolerance and more frequent rebalancing.
    
    Parameters
    ----------
    capital : float
        Initial cash available
    option_params : dict
        Dictionary with option parameters: {'K', 'r', 'sigma', 'T'}
    asset_index : int, optional
        Index of the asset to hedge (default: 0)
    base_tolerance : float, optional
        Base rebalancing tolerance (default: 0.01)
    gamma_sensitivity : float, optional
        Sensitivity to gamma (default: 10.0)
        Higher values make tolerance more responsive to gamma
    """
    
    def __init__(
        self,
        capital,
        option_params,
        asset_index=0,
        base_tolerance=0.01,
        gamma_sensitivity=10.0
    ):
        # Initialize with base tolerance
        super().__init__(capital, option_params, asset_index, base_tolerance)
        self.base_tolerance = base_tolerance
        self.gamma_sensitivity = gamma_sensitivity
    
    def compute_adaptive_tolerance(self, S, t):
        """
        Compute adaptive tolerance based on current gamma.
        
        Formula: tolerance = base_tolerance / (1 + k * gamma)
        where k is the gamma_sensitivity parameter.
        
        Parameters
        ----------
        S : float
            Current stock price
        t : float
            Current time
        
        Returns
        -------
        float
            Adaptive tolerance value
        """
        K = self.option_params['K']
        r = self.option_params['r']
        sigma = self.option_params['sigma']
        T = self.option_params['T']
        
        # Handle edge case: at or past maturity
        if t >= T:
            return self.base_tolerance
        
        gamma = black_scholes_gamma(S, K, r, sigma, T, t)
        
        # Adaptive formula: tighter tolerance when gamma is high
        adaptive_tolerance = self.base_tolerance / (1.0 + self.gamma_sensitivity * gamma)
        
        return adaptive_tolerance
    
    def observe_and_propose(self, card, market_state):
        """
        Observe market state and propose rebalancing trade with adaptive tolerance.
        
        Parameters
        ----------
        card : InformationCard
            Information revealed at this stopping time
        market_state : dict
            Current market state with 'time' and 'prices'
        
        Returns
        -------
        Action
            Proposed rebalancing trade
        """
        S = market_state["prices"][self.asset_index]
        t = market_state["time"]
        
        # Update tolerance based on current gamma
        self.rebalance_tolerance = self.compute_adaptive_tolerance(S, t)
        
        # Use parent class logic with updated tolerance
        return super().observe_and_propose(card, market_state)
