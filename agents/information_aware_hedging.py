from agents.delta_hedging import DeltaHedgingAgent 
from agents.actions import Action 
from options.black_scholes import black_scholes_delta, black_scholes_gamma
from events.cards import InformationCard 
class InformationAwareDeltaHedgingAgent(DeltaHedgingAgent): 
    """ 
    Delta hedging agent that responds to information cards. 
    Uses information to: 
    1. Adjust volatility assumptions (based on realized vol) 
    2. Adjust rebalancing frequency (based on gamma and volatility) 
    3. Optimize transaction costs (based on market microstructure) 
    4. Manage risk (based on realized metrics) 
    """ 
    def __init__( 
        self, 
        capital, 
        option_params, 
        asset_index=0, 
        base_tolerance=0.001, 
        volatility_adaptation=True, 
        gamma_adaptation=True, 
        cost_optimization=True, 
        volatility_adaptation_rate=0.3 
    ): 
        """ 
        Parameters 
        ---------- 
        capital : float 
            Initial capital 
        option_params : dict 
            Option parameters 
        asset_index : int 
            Asset index 
        base_tolerance : float 
            Base rebalancing tolerance 
        volatility_adaptation : bool 
            Whether to adapt volatility based on realized vol 
        gamma_adaptation : bool 
            Whether to adapt tolerance based on gamma 
        cost_optimization : bool 
            Whether to defer rebalancing in high-cost environments 
        volatility_adaptation_rate : float 
            How quickly to adapt volatility (0-1, higher = faster) 
        """ 
        super().__init__(capital, option_params, asset_index, base_tolerance) 
        self.base_tolerance = base_tolerance 
        self.volatility_adaptation = volatility_adaptation 
        self.gamma_adaptation = gamma_adaptation 
        self.cost_optimization = cost_optimization 
        self.volatility_adaptation_rate = volatility_adaptation_rate 
         
        # Store original volatility for reference 
        self.original_sigma = option_params['sigma'] 
        self.current_sigma = option_params['sigma'].copy() if hasattr(option_params['sigma'], 'copy') else option_params['sigma']
         
        # State tracking 
        self.last_volatility_regime = None 
        self.last_gamma_alert = None 
        self.last_microstructure = None 
        self.deferred_rebalancing = False 
     
    def adapt_volatility_from_card(self, card): 
        """Adapt volatility assumption based on volatility regime card""" 
        if not self.volatility_adaptation or card is None:
            return 
        
        # Handle nested payload structure from enhanced cards
        volatility_data = card.payload.get('volatility_regime')
        if volatility_data is None:
            # Fallback: check if card itself is a volatility_regime card
            if card.name == 'volatility_regime':
                volatility_data = card.payload
            else:
                return
        
        realized_vol = volatility_data.get('realized_volatility') 
        if realized_vol is None: 
            return
         
        # Exponential moving average update 
        # New sigma = (1 - α) * old_sigma + α * realized_vol 
        alpha = self.volatility_adaptation_rate 
        self.current_sigma = (1 - alpha) * self.current_sigma + alpha * realized_vol 
         
        # Update option_params 
        self.option_params['sigma'] = self.current_sigma 
     
    def adapt_tolerance_from_cards(self, card): 
        """Adapt rebalancing tolerance based on gamma and volatility""" 
        if card is None:
            return
            
        tolerance_multiplier = 1.0 
        
        # Gamma adaptation - handle nested payload structure
        if self.gamma_adaptation:
            gamma_data = card.payload.get('gamma_alert')
            if gamma_data is None and card.name == 'gamma_alert':
                gamma_data = card.payload
            
            if gamma_data and gamma_data.get('is_high_gamma', False): 
                tolerance_multiplier *= gamma_data.get('tolerance_multiplier', 0.5) 
        
        # Volatility regime adaptation - handle nested payload structure
        volatility_data = card.payload.get('volatility_regime')
        if volatility_data is None and card.name == 'volatility_regime':
            volatility_data = card.payload
            
        if volatility_data:
            vol_multiplier = volatility_data.get('vol_multiplier', 1.0) 
            # Higher volatility → tighter tolerance 
            tolerance_multiplier *= (1.0 / vol_multiplier)
         
        self.rebalance_tolerance = self.base_tolerance * tolerance_multiplier 
     
    def check_cost_optimization(self, card): 
        """Check if rebalancing should be deferred due to high costs""" 
        if not self.cost_optimization or card is None:
            return False 
        
        # Handle nested payload structure
        microstructure_data = card.payload.get('market_microstructure')
        if microstructure_data is None:
            # Fallback: check if card itself is a market_microstructure card
            if card.name == 'market_microstructure':
                microstructure_data = card.payload
            else:
                return False
        
        should_defer = microstructure_data.get('should_defer_rebalancing', False) 
        liquidity = microstructure_data.get('liquidity_score', 1.0)
         
        # Defer if liquidity is very low 
        if should_defer and liquidity < 0.3: 
            return True 
         
        return False 
     
    def observe_and_propose(self, card, market_state): 
        """ 
        Observe market state and information card, propose rebalancing trade. 
         
        This is the key method that makes the agent information-aware. 
        """ 
        S = market_state["prices"][self.asset_index] 
        t = market_state["time"] 
         
        # Process information cards 
        if card is not None: 
            # Adapt volatility based on realized volatility 
            self.adapt_volatility_from_card(card) 
             
            # Adapt tolerance based on gamma and volatility 
            self.adapt_tolerance_from_cards(card) 
             
            # Check if rebalancing should be deferred 
            self.deferred_rebalancing = self.check_cost_optimization(card) 
             
            # Store card information for reference (handle nested structure)
            volatility_data = card.payload.get('volatility_regime')
            if volatility_data or card.name == 'volatility_regime':
                # Create a mock card for storage
                self.last_volatility_regime = InformationCard('volatility_regime', volatility_data or card.payload)
            
            gamma_data = card.payload.get('gamma_alert')
            if gamma_data or card.name == 'gamma_alert':
                self.last_gamma_alert = InformationCard('gamma_alert', gamma_data or card.payload)
            
            microstructure_data = card.payload.get('market_microstructure')
            if microstructure_data or card.name == 'market_microstructure':
                self.last_microstructure = InformationCard('market_microstructure', microstructure_data or card.payload) 
         
        # If rebalancing is deferred, skip trade 
        if self.deferred_rebalancing: 
            return Action(asset_index=self.asset_index, quantity=0.0) 
         
        # Compute target delta using adapted volatility 
        target_delta = self.compute_target_delta(S, t) 
         
        # Get current position 
        current_position = self.positions.get(self.asset_index, 0.0) 
         
        # Compute required trade 
        required_trade = target_delta - current_position 
        # Apply tolerance (which may have been adapted) 
        if abs(required_trade) < self.rebalance_tolerance: 
            required_trade = 0.0 
        return Action( 
            asset_index=self.asset_index, 
            quantity=required_trade 
        ) 
    def get_adaptation_summary(self): 
        """Get summary of how agent has adapted""" 
        return { 
            'original_sigma': self.original_sigma, 
            'current_sigma': self.current_sigma, 
            'sigma_adjustment': (self.current_sigma / self.original_sigma - 1.0) * 100, 
            'current_tolerance': self.rebalance_tolerance, 
            'base_tolerance': self.base_tolerance, 
            'tolerance_adjustment': (self.rebalance_tolerance / self.base_tolerance - 1.0) * 100, 
            'last_volatility_regime': self.last_volatility_regime.payload if self.last_volatility_regime else None, 
            'deferred_rebalancing': self.deferred_rebalancing 
        } 