import torch

class SU2():
    def __init__(self, j):
        self._j = j

    @property
    def j(self):
        return self._j
    
    def operate(self, tensor, alpha, beta, gamma):
        return torch.matmul(self.wigner_D(alpha, beta, gamma), tensor)
    
    def wigner_D(alpha, beta, gamma):
        return