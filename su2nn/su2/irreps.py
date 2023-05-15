from fractions import Fraction

class Irrep():

    def __init__(self, j, p, t):
        if isinstance(j, int, float, Fraction):
            if (2 * j) % 1 == 0:
                self._j = Fraction(int(2 * j), 2)
            else:
                raise ValueError(f'su2.Irreps.j need to have value equivalent to integer or half-integer, but got {j}')
        else:
            raise ValueError(f'su2.Irreps.j require numerical input, but got {type(j)}')

        if p in [-1, 1]:
            self._p = p
        else:
            raise ValueError(f'su2.Irreps.p must be -1 or 1, but got {p}')
        
        if t in [-1, 1]:
            self._t = t
        else:
            raise ValueError(f'su2.Irreps.p must be -1 or 1, but got {t}')

    @property
    def j(self):
        return self._j
    
    @property
    def p(self):
        return self._p
    
    @property
    def t(self):
        return self._t
    
    @property
    def dim(self):
        return int(2 * self.j + 1)
    
    def __repr__(self):
        return f'({self.j},{self.p},{self.t})'