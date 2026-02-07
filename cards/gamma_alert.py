from cards.base import InformationCard


class GammaAlertCard(InformationCard): 
    """ 
    Alert when gamma exposure is high. 
    High gamma means larger hedging errors from discrete rebalancing. 
    Agent should tighten tolerance or rebalance more frequently. 
    """ 
    def __init__(self, gamma_value, is_high=False, threshold=0.015): 
        """ 
        Parameters 
        ---------- 
        gamma_value : float 
            Current gamma exposure 
        is_high : bool 
            Whether gamma exceeds threshold 
        threshold : float 
            Gamma threshold for "high" classification 
        """ 
        payload = { 
            'gamma': gamma_value, 
            'is_high_gamma': is_high, 
            'threshold': threshold, 
            'tolerance_multiplier': 0.5 if is_high else 1.0 
        } 
        super().__init__(name="gamma_alert", payload=payload) 