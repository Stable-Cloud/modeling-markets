# agents/actions.py

class Action:
    """
    Represents a proposed discrete trading action.
    """

    def __init__(self, asset_index, quantity):
        self.asset_index = asset_index
        self.quantity = float(quantity)

    def __repr__(self):
        return f"Action(asset={self.asset_index}, qty={self.quantity})"

