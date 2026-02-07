# execution/costs.py

from abc import ABC, abstractmethod


class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models.
    
    Transaction costs represent the friction of trading in real markets,
    including bid-ask spreads, commissions, and market impact.
    """
    
    @abstractmethod
    def compute_cost(self, quantity, price):
        """
        Compute the transaction cost for a trade.
        
        Parameters
        ----------
        quantity : float
            Number of shares to trade (positive for buy, negative for sell)
        price : float
            Price per share
        
        Returns
        -------
        float
            Transaction cost (always non-negative)
        """
        pass


class ProportionalCostModel(TransactionCostModel):
    """
    Proportional transaction cost model.
    
    Cost is proportional to the notional value of the trade:
    Cost = |quantity| * price * (rate_bps / 10000)
    
    This models bid-ask spreads and percentage-based commissions.
    
    Parameters
    ----------
    rate_bps : float
        Cost rate in basis points (1 bps = 0.01%)
        Example: rate_bps=10 means 0.1% cost per trade
    
    Examples
    --------
    >>> model = ProportionalCostModel(rate_bps=10)  # 0.1% cost
    >>> model.compute_cost(100, 50.0)  # Buy 100 shares at $50
    5.0  # Cost is 0.1% of $5000 = $5
    """
    
    def __init__(self, rate_bps):
        if rate_bps < 0:
            raise ValueError(f"rate_bps must be non-negative. Got: {rate_bps}")
        self.rate_bps = rate_bps
    
    def compute_cost(self, quantity, price):
        """
        Compute proportional transaction cost.
        
        Parameters
        ----------
        quantity : float
            Number of shares to trade
        price : float
            Price per share
        
        Returns
        -------
        float
            Transaction cost
        """
        notional = abs(quantity) * price
        cost = notional * (self.rate_bps / 10000.0)
        return cost
    
    def __repr__(self):
        return f"ProportionalCostModel(rate_bps={self.rate_bps})"


class FixedPlusProportionalCostModel(TransactionCostModel):
    """
    Fixed plus proportional transaction cost model.
    
    Cost = fixed_cost + |quantity| * price * (rate_bps / 10000)
    
    This models a flat commission plus a percentage fee.
    Common in retail brokerage accounts.
    
    Parameters
    ----------
    fixed_cost : float
        Flat fee per trade (in currency units)
    rate_bps : float
        Proportional cost rate in basis points
    
    Examples
    --------
    >>> model = FixedPlusProportionalCostModel(fixed_cost=1.0, rate_bps=5)
    >>> model.compute_cost(100, 50.0)  # Buy 100 shares at $50
    3.5  # $1 fixed + 0.05% of $5000 = $1 + $2.50
    """
    
    def __init__(self, fixed_cost, rate_bps):
        if fixed_cost < 0:
            raise ValueError(f"fixed_cost must be non-negative. Got: {fixed_cost}")
        if rate_bps < 0:
            raise ValueError(f"rate_bps must be non-negative. Got: {rate_bps}")
        self.fixed_cost = fixed_cost
        self.rate_bps = rate_bps
    
    def compute_cost(self, quantity, price):
        """
        Compute fixed plus proportional transaction cost.
        
        Parameters
        ----------
        quantity : float
            Number of shares to trade
        price : float
            Price per share
        
        Returns
        -------
        float
            Transaction cost
        """
        # No cost if no trade
        if quantity == 0:
            return 0.0
        
        notional = abs(quantity) * price
        proportional_cost = notional * (self.rate_bps / 10000.0)
        total_cost = self.fixed_cost + proportional_cost
        return total_cost
    
    def __repr__(self):
        return f"FixedPlusProportionalCostModel(fixed_cost={self.fixed_cost}, rate_bps={self.rate_bps})"


class NonlinearCostModel(TransactionCostModel):
    """
    Nonlinear transaction cost model with market impact.
    
    Cost = |quantity|^impact_exponent * price * (rate_bps / 10000)
    
    This models market impact: larger trades have disproportionately
    higher costs due to moving the market.
    
    Parameters
    ----------
    rate_bps : float
        Base cost rate in basis points
    impact_exponent : float, optional
        Exponent for market impact (default: 1.5)
        - impact_exponent = 1.0: linear (same as proportional)
        - impact_exponent > 1.0: superlinear (market impact)
        - Typical values: 1.2 to 1.8
    
    Examples
    --------
    >>> model = NonlinearCostModel(rate_bps=10, impact_exponent=1.5)
    >>> model.compute_cost(100, 50.0)
    50.0  # 100^1.5 * 50 * 0.001 = 1000 * 50 * 0.001
    """
    
    def __init__(self, rate_bps, impact_exponent=1.5):
        if rate_bps < 0:
            raise ValueError(f"rate_bps must be non-negative. Got: {rate_bps}")
        if impact_exponent < 1.0:
            raise ValueError(f"impact_exponent must be >= 1.0. Got: {impact_exponent}")
        self.rate_bps = rate_bps
        self.impact_exponent = impact_exponent
    
    def compute_cost(self, quantity, price):
        """
        Compute nonlinear transaction cost with market impact.
        
        Parameters
        ----------
        quantity : float
            Number of shares to trade
        price : float
            Price per share
        
        Returns
        -------
        float
            Transaction cost
        """
        abs_quantity = abs(quantity)
        if abs_quantity == 0:
            return 0.0
        
        # Nonlinear scaling of quantity
        scaled_quantity = abs_quantity ** self.impact_exponent
        cost = scaled_quantity * price * (self.rate_bps / 10000.0)
        return cost
    
    def __repr__(self):
        return f"NonlinearCostModel(rate_bps={self.rate_bps}, impact_exponent={self.impact_exponent})"


class TieredCostModel(TransactionCostModel):
    """
    Tiered transaction cost model with volume discounts.
    
    Different cost rates apply based on trade size tiers.
    Larger trades get lower per-unit costs (volume discounts).
    
    Parameters
    ----------
    tiers : list of tuples
        List of (threshold, rate_bps) tuples, sorted by threshold
        Example: [(0, 10), (1000, 5), (10000, 2)]
        - Trades up to 1000 shares: 10 bps
        - Trades 1000-10000 shares: 5 bps
        - Trades over 10000 shares: 2 bps
    
    Examples
    --------
    >>> tiers = [(0, 10), (1000, 5), (10000, 2)]
    >>> model = TieredCostModel(tiers)
    >>> model.compute_cost(500, 100.0)  # 500 shares at $100
    50.0  # 10 bps on $50,000
    >>> model.compute_cost(5000, 100.0)  # 5000 shares at $100
    250.0  # 5 bps on $500,000
    """
    
    def __init__(self, tiers):
        if not tiers:
            raise ValueError("tiers must be non-empty")
        
        # Sort tiers by threshold
        self.tiers = sorted(tiers, key=lambda x: x[0])
        
        # Validate
        for threshold, rate_bps in self.tiers:
            if threshold < 0:
                raise ValueError(f"Threshold must be non-negative. Got: {threshold}")
            if rate_bps < 0:
                raise ValueError(f"rate_bps must be non-negative. Got: {rate_bps}")
    
    def compute_cost(self, quantity, price):
        """
        Compute tiered transaction cost.
        
        Parameters
        ----------
        quantity : float
            Number of shares to trade
        price : float
            Price per share
        
        Returns
        -------
        float
            Transaction cost
        """
        abs_quantity = abs(quantity)
        
        # Find applicable tier
        applicable_rate = self.tiers[0][1]  # Default to first tier
        for threshold, rate_bps in reversed(self.tiers):
            if abs_quantity >= threshold:
                applicable_rate = rate_bps
                break
        
        notional = abs_quantity * price
        cost = notional * (applicable_rate / 10000.0)
        return cost
    
    def __repr__(self):
        return f"TieredCostModel(tiers={self.tiers})"
