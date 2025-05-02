
class Security(): # class to store information about a security
    def __init__(self, name: str):
        self.value = 0 # value of the security (will be calculated as current price * number of shares owned)
        self.shares = 0 # number of shares owned
        self.name = name # name/ticker for the security as a string
        self.tradeable = False # is the security tradeable at the current time step?
    def __repr__(self):
        return self.name