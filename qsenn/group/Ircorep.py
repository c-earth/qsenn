import abc
import torch

from qsenn.group.Irrep import Irrep

class IrcorepInterface(Irrep):
    @property
    @abc.abstractmethod
    def G(self):
        pass

class Ircorep(IrcorepInterface):
    def __init__(self, ircorep):
        if isinstance(ircorep, Ircorep):
            return ircorep
        
        if isinstance(ircorep, tuple) and len(ircorep) == len(repr(self.G).split(',')) + 1 and isinstance(ircorep[-1], int):
            labels = (int(l) for l in ircorep)
        elif isinstance(ircorep, str) and len(ircorep.split(',')) == len(repr(self.G).split(',')) + 1:
            labels = (int(l) for l in ircorep.split(','))
        else:
            raise ValueError(f'unable to interpret {ircorep} as Ircorep with \
                             {self.G} as unitary subgroup\'s Irrep')

        self._labels = labels
        self._irrep = self.G(self.labels[:-1])

    @property
    def labels(self):
        return self._labels

    @property
    def irrep(self):
        return self._irrep

    def __repr__(self):
        return ','.join(f'{l}' for l in self.labels)
    
class IrcorepEquivInterface(Ircorep):
    @property
    @abc.abstractmethod
    def U(self):
        pass

    @property
    @abc.abstractmethod
    def UUast(self):
        pass

class IrcorepEquiv(IrcorepEquivInterface):
    @property
    def t(self):
        return self.labels[-1]
    
    @property
    def lamb(self):
        return self.UUast / self.t
    
    @property
    def E(self):
        if self.lamb == 1:
            return torch.tensor([[1]])
        elif self.lamb == -1:
            return torch.tensor([[0, -1], [1, 0]])
        else:
            raise ValueError('lambda parameter must be an interger -1 or 1')
    
    @property
    def dim(self):
        if self.lamb == 1:
            return self.irrep.dim
        elif self.lamb == -1:
            return 2 * self.irrep.dim
        else:
            raise ValueError('lambda parameter must be an interger -1 or 1')

    def D(self, *args):
        tau = args[-1]
        DG = self.irrep.D(*args[:-1])
        Dhalf = (self.UUast ** (tau // 2)) * torch.matmul(DG, torch.matrix_power(self.U, tau % 2))
        return torch.kron(torch.matrix_power(self.E, tau % 4), Dhalf)

    def operate(self, target, *args):
        if args[-1] % 2 == 1:
            return torch.matmul(self.D(*args), torch.conj(target))
        else:
            return torch.matmul(self.D(*args), target)

    @classmethod
    def iterator(cls, *labels_max):
        _ = labels_max[-1]
        for irrep in cls.G.iterator(*labels_max[:-1]):
            for t in [-1, 1]:
                ircorepEquiv = (int(l) for l in repr(irrep).split(',')) + (t,)
                yield ircorepEquiv(ircorepEquiv)