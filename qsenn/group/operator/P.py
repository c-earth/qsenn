import torch

c128 = torch.complex128

class D_P():
    def __init__(self, p):
        self._p = p
        
    @property
    def p(self):
        return self._p
    
    def __repr__(self):
        return f'D_P^({self.p})'
    
    def op(self, rho):
        return torch.tensor(self.p ** rho, dtype = c128)

    def operate(self, target, rho):
        operator = self.op(rho)
        return torch.mul(operator, target)
    
    @classmethod
    def tp(cls, D_P1, D_P2):
        return D_P(D_P1.p * D_P2.p)