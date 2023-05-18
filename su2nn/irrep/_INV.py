class INV():
    def __init__(self, p):
        if p not in [-1, 1]:
            raise ValueError(f'p must be -1 or 1, but p = {p} is given.')
        self._p = p

    @property
    def p(self):
        return self._p
    
    def __mul__(self, other):
        return self.p * other