import torch
import itertools

from qsenn.group.Irrep import Irrep

class IrrepTP(Irrep):
    @property
    def parents(self):
        return tuple()

    def __init__(self, irrepTP):
        if isinstance(irrepTP, IrrepTP):
            return irrepTP
        
        if isinstance(irrepTP, tuple) and len(irrepTP) == len(self.parents) and isinstance(irrepTP[-1], int):
            labels = (int(l) for l in irrepTP)
        elif isinstance(irrepTP, str) and len(irrepTP.split(',')) == len(self.parents):
            labels = (int(l) for l in irrepTP.split(','))
        else:
            raise ValueError(f'unable to interpret {irrepTP} as IrrepTP with \
                             {self.parents} as parents\' Irrep\'s')

        self._labels = labels
        self._comps = (parent((l,)) for parent, l in zip(self.parents, labels))

    @property
    def dim(self):
        prod = 1
        for comp in self.comps:
            prod *= comp.dim
        return prod
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def comps(self):
        return self._comps

    def __repr__(self):
        return ','.join(f'{l}' for l in self.labels)
    
    def __mul__(self, other):
        if isinstance(other, IrrepTP):
            if self.parents == other.parents:
                labelss = []
                for comp1, comp2 in zip(self.comps, other.comps):
                    labelss.append((int(repr(irrep)) for irrep in comp1 * comp2))
                for irrepTP in itertools.product(*labelss):
                    yield IrrepTP(tuple(irrepTP))
                    
            else:
                raise ValueError(f'multiplication must be between IrrepTP with the same parents, \
                                 but got {self.parents}, and {other.parents}')
        else:
            raise ValueError(f'multiplication must be between IrrepTP, but got {type(other)}')
    
    def D(self, *args):
        if len(args) != len(self.parents):
            raise ValueError(f'need {len(self.parents)} arguments, but {len(args)} are given')
        Dkron = torch.tensor([[1]])
        for comp, arg in zip(self.comps, args):
            Dkron = torch.kron(Dkron, comp.D(arg))
        return Dkron
    
    def operate(self, target, *args):
        return torch.matmul(self.D(*args), target)

    @classmethod
    def iterator(cls, *labels_max):
        labelss = []
        for parent, label_max in zip(cls.parents, labels_max):
            labelss.append((int(repr(irrep)) for irrep in parent.iterator(label_max)))
        for irrepTP in itertools.product(*labelss):
            yield IrrepTP(tuple(irrepTP))