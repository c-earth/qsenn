import torch

from qsenn.group.Ircorep import IrcorepEquiv
from qsenn.group.SU2xP import IrrepSU2xP

class IrcorepSU2xPxT(IrcorepEquiv):
    @property
    def G(self):
        return IrrepSU2xP
    
    @property
    def U(self):
        _2ms = torch.arange(-self._2j, self._2j + 1, 2)
        return torch.fliplr(torch.diag(torch.pow(1j, _2ms)))
    
    @property
    def UUast(self):
        return (-1) ** (self._2j)
    
    @property
    def j(self):
        return self._2j/2
    
    @property
    def _2j(self):
        return self.labels[0]
    
    @property
    def p(self):
        return self.labels[1]

    def __mul__(self, other):
        if isinstance(other, IrcorepEquiv):
            if self.G == other.G:
                loop = 1
                if self.lamb == -1 and other.lamb == -1:
                    loop = 4
                t = self.t * other.t
                for _ in range(loop):
                    for irrep in self.irrep * other.irrep:
                        yield IrcorepSU2xPxT(irrep.labels + (t,))
                    
            else:
                raise ValueError(f'multiplication must be between IrcorepEquiv with the same unitary subgroup, \
                                 but got {self.G}, and {other.G}')
        else:
            raise ValueError(f'multiplication must be between IrcorepEquiv, but got {type(other)}')