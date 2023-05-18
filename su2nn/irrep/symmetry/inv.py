import torch

class INV():
    def __init__(self, p):
        self._p = p

    @property
    def p(self):
        return self._p
    
    def operate(self, tensor):
        return torch.mul(self.p, tensor)
    
    @classmethod
    def from_irrep(cls, irrep):
        return cls(irrep.p)