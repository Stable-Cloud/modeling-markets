# events/base.py

class InformationCard:
    """
    Discrete packet of information revealed at a stopping time.
    """

    def __init__(self, name, payload):
        """
        Parameters
        ----------
        name : str
            Human-readable identifier for the card.
        payload : dict
            Arbitrary information content.
        """
        self.name = name
        self.payload = payload

    def __repr__(self):
        return f"InformationCard(name={self.name})"
