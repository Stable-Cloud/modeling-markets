class EventEngine: 
    """ 
    Drives the pause–inspect–resume loop. 
    Enhanced to generate information cards dynamically. 
    """ 
    def __init__(self, market, stopping_events, information_generator=None): 
        """ 
        Parameters 
        ---------- 
        market : MarketSubstrate 
            Market instance 
        stopping_events : list 
            List of (StoppingCondition, InformationCard) tuples 
        information_generator : InformationGenerator, optional 
            Generator for dynamic information cards 
        """ 
        self.market = market 
        self.stopping_events = stopping_events 
        self.information_generator = information_generator 
    def step_until_event(self, dt): 
        """ 
        Advance the market until any stopping condition triggers. 
        Returns the associated InformationCard, enhanced with dynamic information. 
        """ 
        while True: 
            self.market.advance_time(dt) 
            state = self.market.get_state() 
            for condition, card in self.stopping_events: 
                if condition.has_triggered(state): 
                    # Enhance card with dynamic information if generator exists 
                    if self.information_generator is not None: 
                        enhanced_card = self._enhance_card(card, state) 
                        return enhanced_card 
                    return card 
    def _enhance_card(self, base_card, market_state): 
        """Enhance base card with dynamic information""" 
        # For now, add realized metrics 
        # Can be extended to add multiple cards or combine information 
        if self.information_generator is None: 
            return base_card 
        # Generate additional information cards 
        realized_metrics = self.information_generator.generate_realized_metrics_card(market_state) 
        gamma_alert = self.information_generator.generate_gamma_alert_card(market_state) 
        volatility_regime = self.information_generator.generate_volatility_regime_card(market_state) 
        microstructure = self.information_generator.generate_market_microstructure_card(market_state) 
        # Combine into base card payload (or create composite card) 
        enhanced_payload = base_card.payload.copy() 
        if realized_metrics: 
            enhanced_payload['realized_metrics'] = realized_metrics.payload 
        if gamma_alert: 
            enhanced_payload['gamma_alert'] = gamma_alert.payload 
        if volatility_regime: 
            enhanced_payload['volatility_regime'] = volatility_regime.payload 
        if microstructure: 
            enhanced_payload['market_microstructure'] = microstructure.payload 
        # Create enhanced card 
        from events.cards import InformationCard 
        return InformationCard( 
            name=base_card.name, 
            payload=enhanced_payload 
        )