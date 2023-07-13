import torch

from qsenn.group.Irrep import Irrep

class IrrepSU2(Irrep):
    def __init__(self, irrepSU2):
        if isinstance(irrepSU2, IrrepSU2):
            return irrepSU2
        
        if isinstance(irrepSU2, tuple) and len(irrepSU2) == 1 and isinstance(irrepSU2[-1], int):
            twoj = int(irrepSU2[0])
        elif isinstance(irrepSU2, str) and len(irrepSU2.split(',')) == 1:
            twoj = int(irrepSU2)
        else:
            raise ValueError(f'unable to interpret {irrepSU2} as IrrepSU2')

        self._2j = twoj

    @property
    def dim(self):
        return self._2j + 1
    
    @property
    def j(self):
        return self._2j/2

    def __repr__(self):
        return f'{self._2j}'
    
    def __mul__(self, other):
        if isinstance(other, IrrepSU2):
            for twoj in range(abs(self._2j - other._2j), self._2j + other._2j + 1, 2):
                irrepSU2 = (twoj,)
                yield IrrepSU2(irrepSU2)
        else:
            raise ValueError(f'multiplication must be between IrrepSU2, but got {type(other)}')
    
    def D(self, phi):
        Jx, Jy, Jz = self.generators()
        phix, phiy, phiz = phi
        exponent = -1j*(Jx*phix + Jy*phiy + Jz*phiz)
        return torch.matrix_exp(exponent)
    
    def operate(self, target, phi):
        return torch.matmul(self.D(phi), target)

    @classmethod
    def iterator(cls, twoj_max):
        for twoj in range(0, twoj_max+1):
            irrepSU2 = (twoj,)
            yield IrrepSU2(irrepSU2)

    @staticmethod
    def generators(self):
        js = torch.arange(-self._2j, self._2j + 1, 2)/2

        Jp = torch.diag(torch.sqrt(self.j * (self.j + 1) - js[:-1] * (js[:-1] + 1)), diagonal = -1)
        Jm = torch.diag(torch.sqrt(self.j * (self.j + 1) - js[1:] * (js[1:] - 1)), diagonal = 1)

        Jx = (Jp + Jm)/2
        Jy = (Jp - Jm)/2j
        Jz = torch.diag(js)
        return Jx, Jy, Jz