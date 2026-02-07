from cards.base import InformationCard


class MarketMicrostructureCard(InformationCard): 
    """ 
    Information about market microstructure (bid-ask spreads, liquidity). 
    Helps optimize execution and transaction costs. 
    """ 
    def __init__(self, bid_ask_spread, liquidity_score, transaction_cost_estimate): 
        """ 
        Parameters 
        ---------- 
        bid_ask_spread : float 
            Current bid-ask spread (as percentage) 
        liquidity_score : float 
            Liquidity score (0-1, higher = more liquid) 
        transaction_cost_estimate : float 
            Estimated transaction cost in bps 
        """ 
        payload = { 
            'bid_ask_spread': bid_ask_spread, 
            'liquidity_score': liquidity_score, 
            'txn_cost_bps': transaction_cost_estimate, 
            'should_defer_rebalancing': liquidity_score < 0.3 
        } 
        super().__init__(name="market_microstructure", payload=payload) 