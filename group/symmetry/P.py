import torch

class D_P():
    def __init__(self, p):
        if p in [-1, 1]:
            self._p = p
        else:
            raise ValueError(f'Parity must be an integer -1 or 1, but {p} is given.')
        
    @property
    def p(self):
        return self._p
    
    def __repr__(self):
        return f'D_P^({self.p})'

    def operate(self, target, rho):
        if not isinstance(target, torch.Tensor):
            raise ValueError(f'The target of operation must be torch.Tensor, but {type(target)} is given')
        
        if not isinstance(rho, int):
            raise ValueError(f'The number of parity (rho) operations must be integer, but {type(rho)} is given')
        
        return torch.mul(torch.tensor(self.p ** rho, dtype = target.dtype), target)
    
    @classmethod
    def tp(cls, D_P1, D_P2):
        if not isinstance(D_P1, D_P):
            raise ValueError(f'Tensor product arguments must be a D_P, but {type(D_P1)} is given for the first argument.')
        
        if not isinstance(D_P2, D_P):
            raise ValueError(f'Tensor product arguments must be a D_P, but {type(D_P2)} is given for the second argument.')
        
        return D_P(D_P1.p * D_P2.p)