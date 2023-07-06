import torch

c128 = torch.complex128

class D_SU2():
    def __init__(self, j):
        self._j = j
        self._J = self.SU2_generators()

    @property
    def j(self):
        return self._j
    
    @property
    def J(self):
        return self._J
    
    def __repr__(self):
        return f'D_SU2^({self.j})'
    
    def SU2_generators(self):
        Jp = torch.diag(torch.tensor([(self.j*(self.j + 1) - (a - self.j)*(a - self.j - 1)) ** 0.5 for a in range(1, int(2*self.j + 1))], dtype = c128))
        Jn = Jp.T
        Jx = (Jp + Jn)/2
        Jy = -1j*(Jp - Jn)/2
        Jz = torch.diag(torch.tensor([a - self.j for a in range(int(2*self.j + 1))], dtype = c128))
        return Jx, Jy, Jz
    
    def op(self, phi):
        return torch.linalg.matrix_exp(-1j*(self.J[0]*phi[0] + self.J[1]*phi[1] + self.J[2]*phi[2]))

    def operate(self, target, phi):
        operator = self.op(phi)
        return torch.matmul(operator, target)