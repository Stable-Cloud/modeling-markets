import numpy as np 
from events.cards.volatility_regime import VolatilityRegimeCard 
from events.cards.gamma_alert import GammaAlertCard 
from events.cards.market_microstructure import MarketMicrostructureCard 
from events.cards.realized_metrics import RealizedMetricsCard 
from options.black_scholes import black_scholes_gamma 
class InformationGenerator: 
    """ 
    Generates information cards based on current and historical market state. 
    This is NOT predictive - it computes observable metrics from price history. 
    """ 
    def __init__(self, option_params, lookback_window=20, gamma_threshold=0.015): 
        """ 
        Parameters 
        ---------- 
        option_params : dict 
            Option parameters {'K', 'r', 'sigma', 'T'} 
        lookback_window : int 
            Number of steps to look back for realized metrics 
        gamma_threshold : float 
            Gamma threshold for high gamma alerts 
        """ 
        self.option_params = option_params 
        self.lookback_window = lookback_window 
        self.gamma_threshold = gamma_threshold 
        self.price_history = [] 
        self.time_history = [] 
     
    def update_history(self, market_state): 
        """Update price and time history""" 
        S = market_state['prices'][0] 
        t = market_state['time'] 
        self.price_history.append(S) 
        self.time_history.append(t) 
         
        # Keep only recent history 
        if len(self.price_history) > self.lookback_window * 2: 
            self.price_history = self.price_history[-self.lookback_window:] 
            self.time_history = self.time_history[-self.lookback_window:] 
     
    def compute_realized_volatility(self): 
        """Compute realized volatility from price history""" 
        if len(self.price_history) < 2: 
            return None 
         
        prices = np.array(self.price_history) 
        times = np.array(self.time_history) 
         
        # Compute log returns 
        log_returns = np.diff(np.log(prices)) 
         
        # Compute time intervals 
        dt_values = np.diff(times) 
         
        # Annualized volatility 
        if len(log_returns) > 1 and np.mean(dt_values) > 0: 
            var_returns = np.var(log_returns) 
            avg_dt = np.mean(dt_values) 
            realized_vol = np.sqrt(var_returns / avg_dt) 
            return realized_vol 
         
        return None 
     
    def classify_volatility_regime(self, realized_vol, assumed_vol=0.2): 
        """Classify volatility regime based on realized vs assumed""" 
        if realized_vol is None: 
            return 'normal', 1.0 
         
        ratio = realized_vol / assumed_vol 
         
        if ratio < 0.7: 
            return 'low', min(1.0, 0.5 + ratio) 
        elif ratio > 1.3: 
            return 'high', min(1.0, 0.5 + (2 - ratio)) 
        else: 
            return 'normal', 0.8 
     
    def generate_volatility_regime_card(self, market_state): 
        """Generate volatility regime card""" 
        self.update_history(market_state) 
        realized_vol = self.compute_realized_volatility() 
         
        if realized_vol is None: 
            return None 
         
        regime, confidence = self.classify_volatility_regime( 
            realized_vol,  
            self.option_params.get('sigma', 0.2) 
        ) 
         
        return VolatilityRegimeCard(regime, realized_vol, confidence) 
     
    def generate_gamma_alert_card(self, market_state): 
        """Generate gamma alert card""" 
        S = market_state['prices'][0] 
        t = market_state['time'] 
        K = self.option_params['K'] 
        r = self.option_params['r'] 
        sigma = self.option_params['sigma'] 
        T = self.option_params['T'] 
         
        if t >= T: 
            return None 
         
        gamma = black_scholes_gamma(S, K, r, sigma, T, t) 
        is_high = gamma > self.gamma_threshold 
         
        return GammaAlertCard(gamma, is_high, self.gamma_threshold) 
     
    def generate_market_microstructure_card(self, market_state): 
        """Generate market microstructure card""" 
        # Synthetic for now - in real implementation, would come from market data 
        # Can be made stochastic but independent of price dynamics 
         
        # Simulate bid-ask spread (independent of price level) 
        base_spread = 0.001  # 10 bps 
        spread_variation = np.random.uniform(0.8, 1.2) 
        bid_ask_spread = base_spread * spread_variation 
         
        # Liquidity score (higher when spread is lower) 
        liquidity_score = 1.0 / (1.0 + bid_ask_spread * 100) 
         
        # Transaction cost estimate 
        txn_cost_bps = bid_ask_spread * 10000 
        return MarketMicrostructureCard( 
            bid_ask_spread,  
            liquidity_score,  
            txn_cost_bps 
        ) 
    def generate_realized_metrics_card(self, market_state): 
        """Generate realized metrics card""" 
        self.update_history(market_state) 
        realized_vol = self.compute_realized_volatility() 
        if realized_vol is None or len(self.price_history) < 2: 
            return None 
        prices = np.array(self.price_history) 
        times = np.array(self.time_history) 
        # Realized drift 
        returns = np.diff(prices) / prices[:-1] 
        dt_avg = np.mean(np.diff(times)) 
        realized_drift = np.mean(returns) / dt_avg if dt_avg > 0 else 0.0 
        # Momentum (recent price change) 
        if len(prices) >= 5: 
            momentum = (prices[-1] - prices[-5]) / prices[-5] 
        else: 
            momentum = 0.0 
        return RealizedMetricsCard(realized_vol, realized_drift, momentum) 