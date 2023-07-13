import torch

from qsenn.group.G import Irrep

class IrrepP(Irrep):
    def __init__(self, irrepP):
        if isinstance(irrepP, IrrepP):
            return irrepP
        
        if isinstance(irrepP, tuple) and len(irrepP) == 1 and isinstance(irrepP[-1], int):
            p = int(irrepP[0])
        elif isinstance(irrepP, str):
            p = int(irrepP)
        else:
            raise ValueError(f'unable to interpret {irrepP} as IrrepP')

        self._p = p

    @property
    def dim(self):
        return 1
    
    @property
    def p(self):
        return self._p

    def __repr__(self):
        return f'{self.p}'
    
    def __mul__(self, other):
        if isinstance(other, IrrepP):
            p = self.p * other.p
            irrepP = (p,)
            yield IrrepP(irrepP)
        else:
            raise ValueError(f'multiplication must be between IrrepP, but got {type(other)}')
    
    def D(self, rho):
        if not isinstance(rho, int):
            raise ValueError(f'parity is parameterized by integer, but got {type(rho)}')
        return self.p ** (rho % 2)
    
    def operate(self, target, rho):
        return torch.matmul(self.D(rho), target)

    @classmethod
    def iterator(cls):
        for p in [-1, 1]:
            irrepP = (p,)
            yield IrrepP(irrepP)