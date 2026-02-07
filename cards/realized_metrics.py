from cards.base import InformationCard


class RealizedMetricsCard(InformationCard): 
    """ 
    Realized metrics computed from recent price history. 
    These are backward-looking (observable), not predictive. 
    """ 
    def __init__(self, realized_vol, realized_drift, price_momentum): 
        """ 
        Parameters 
        ---------- 
        realized_vol : float 
            Realized volatility over recent window 
        realized_drift : float 
            Realized drift (average return) 
        price_momentum : float 
            Recent price momentum indicator 
        """ 
        payload = { 
            'realized_volatility': realized_vol, 
            'realized_drift': realized_drift, 
            'momentum': price_momentum, 
            'volatility_ratio': realized_vol / 0.2  # Ratio to assumed vol 
        } 
        super().__init__(name="realized_metrics", payload=payload) 