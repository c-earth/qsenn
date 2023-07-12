from G import Irrep

p2n = {'o': -1, 'e': 1}
n2p = {-1: 'o', 1: 'e'}

class IrrepP(Irrep):
    def __init__(self, p):
        if isinstance(p, IrrepP):
            p = p.p
        elif p in p2n:
            p = p2n[p]

        if p not in n2p:
            raise ValueError(f'parity must be an integer -1, +1, or a string \'o\', \'e\', but got {p}')
        else:
            self._p = p