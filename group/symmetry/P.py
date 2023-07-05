import torch

p2num = {'o': -1, 'e': 1}
num2p = {-1: 'o', 1: 'e'}

class D_P():
    def __init__(self, p):
        if isinstance(p, str) and p in p2num:
            p = p2num[p]
        
        if p in num2p:
            self.p = p
        else:
            raise ValueError(f'Parity must be an integer -1 or 1, or a string \'o\' or \'e\', but {p} is given.')

    def operate(self, target, rho):
        if not isinstance(target, torch.Tensor):
            raise ValueError(f'The target of operation must be torch.Tensor, but {type(target)} is given')
        
        if not isinstance(rho, int):
            raise ValueError(f'The number of parity (rho) operations must be integer, but {type(rho)} is given')
        
        return torch.mul(torch.tensor(self.p ** rho, dtype = target.dtype), target)