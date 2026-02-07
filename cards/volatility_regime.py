from cards.base import InformationCard


class VolatilityRegimeCard(InformationCard): 
    """ 
    Information about current volatility regime. 
    This is derived from realized volatility, not predictive. 
    Helps agent adapt to observed market conditions. 
    """ 
    def __init__(self, regime_type, realized_vol, confidence=1.0): 
        """ 
        Parameters 
        ---------- 
        regime_type : str 
            'low', 'normal', 'high' 
        realized_vol : float 
            Realized volatility over recent window 
        confidence : float 
            Confidence in regime classification (0-1) 
        """ 
        payload = {
            'regime': regime_type,
            'realized_volatility': realized_vol,
            'confidence': confidence,
            'vol_multiplier': self._compute_multiplier(regime_type)
        }
        super().__init__(name="volatility_regime", payload=payload)
    
    def _compute_multiplier(self, regime): 
        """Volatility multiplier for hedging frequency adjustment""" 
        multipliers = {'low': 0.7, 'normal': 1.0, 'high': 1.5} 
        return multipliers.get(regime, 1.0) 