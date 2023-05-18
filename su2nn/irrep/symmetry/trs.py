import torch

class TRS():
    def __init__(self, t):
        self._t = t

    @property
    def t(self):
        return self._t
    
    def operate(self, tensor, n):
        out = tensor
        for _ in range(n):
            out = torch.mul(self.t, torch.conj(torch.flip(tensor)))
        return out