import torch

from python_project.qsenn.qsenn.group.G import Irrep

p2n = {'o': -1, 'e': 1}
n2p = {-1: 'o', 1: 'e'}

class IrrepP():
    def __init__(self, p):
        if isinstance(p, IrrepP):
            p = p.p
        elif p in p2n:
            p = p2n[p]

        if p not in n2p:
            raise ValueError(f'parity must be an integer -1, +1, or a string \'o\', \'e\', but got {p}')
        else:
            self._p = p

    @property
    def p(self):
        return self._p
    
    @property
    def dim(self):
        return 1
    
    def __repr__(self):
        return f'{n2p[self.p]}'
    
    def __mul__(self, other):
        other = IrrepP(other)
        p = self.p * other.p
        yield IrrepP(p)

    def __rmul__(self, other):
        if not isinstance(other, int):
            raise ValueError(f'multiplicity must be an integer, but got {type(other)}')
        elif other <= 0:
            raise ValueError(f'multiplicity must be positive, but got {other}')
        return IrrepsP([(other, self)])
    
    def __add__(self, other):
        return IrrepsP(self) + IrrepsP(other)
    
    def is_scalar(self):
        return self.p == 1
    
    def D(self, rho):
        if not isinstance(rho, int):
            raise ValueError(f'parity parameterization must be an integer, but got {type(rho)}')
        elif rho < 0:
            raise ValueError(f'parity parameterization must be non-negative, but got {rho}')
        return torch.tensor([[self.p ** rho]])
    
    @classmethod
    def iterator(cls):
        for p in n2p:
            yield IrrepP(p)