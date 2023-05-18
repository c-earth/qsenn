import torch

class TRS():
    def __init__(self, t):
        self._t = t

    @property
    def t(self):
        return self._t
    
    def operate(self, tensor):
        return torch.mul(self.t, torch.conj(torch.flip(tensor)))
    
    @classmethod
    def from_irrep(cls, irrep):
        return cls(irrep.t)