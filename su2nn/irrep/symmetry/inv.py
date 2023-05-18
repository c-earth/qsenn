import torch

class INV():
    def __init__(self, p):
        self._p = p

    @property
    def p(self):
        return self._p
    
    def operate(self, tensor, n):
        out = tensor
        for _ in range(n):
            out = torch.mul(self.p, tensor)
        return out