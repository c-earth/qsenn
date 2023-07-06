import torch

c128 = torch.complex128

class D_SU2():
    def __init__(self, j):
        self._j = j
        self._J = D_SU2.SU2_generators(self.j)

    @property
    def j(self):
        return self._j
    
    @property
    def J(self):
        return self._J
    
    def __repr__(self):
        return f'D_SU2^({self.j})'
    
    def op(self, phi):
        return torch.linalg.matrix_exp(-1j*torch.einsum('ijk,i->jk', self.J, phi))

    def operate(self, target, phi):
        operator = self.op(phi)
        return torch.matmul(operator, target)
    
    @staticmethod
    def SU2_generators(j):
        Jp = torch.diag(torch.tensor([(j*(j + 1) - (a - j + 1)*(a - j)) ** 0.5 for a in range(int(2*j))], dtype = c128))
        Jn = Jp.T
        Jx = (Jp + Jn)/2
        Jy = -1j*(Jp - Jn)/2
        Jz = torch.diag(torch.tensor([a - j for a in range(int(2*j + 1))], dtype = c128))
        return torch.tensor([Jx, Jy, Jz], dtype = c128)