import torch

from SU2 import D_SU2
from P import D_P

c128 = torch.complex128

class D_SU2xPxT():
    def __init__(self, j, p, t):
        self._j = j
        self._p = p
        self._t = t

        self._D_SU2 = D_SU2(j)
        self._D_P = D_P(p)

        self._E = self.generate_E()
        self._U = self.generate_U()

    @property
    def j(self):
        return self._j
    
    @property
    def p(self):
        return self._p

    @property
    def t(self):
        return self._t
    
    @property
    def D_SU2(self):
        return self._D_SU2
    
    @property
    def D_P(self):
        return self._D_P
    
    @property
    def E(self):
        return self._E
    
    @property
    def U(self):
        return self._U
    
    def __repr__(self):
        return f'D_SU2xPxT^({self.j}, {self.p}, {self.t})'
    
    def generate_E(self):
        if self.t == (-1) ** int(2*self.j):
            return torch.tensor([1], dtype = c128)
        else:
            return torch.tensor([[0, -1], [1, 0]], dtype = c128)
    
    def generate_U(self):
        anti_diagonal = torch.tensor([1j ** int(2*(a - self.j)) for a in range(int(2*self.j + 1))], dtype = c128)
        return torch.fliplr(torch.diag(anti_diagonal))
    
    def op(self, phi, rho, tau):
        SU2_op = self.D_SU2.op(phi)
        P_op = self.D_P.op(rho)
        SU2xP_op = torch.kron(SU2_op, P_op)
        
        E_tau = torch.matrix_power(self.E, tau%4)
        U_tau = ((-1) ** int(2*self.j*(tau//2))) * (self.U ** (tau%2))
        return torch.kron(E_tau, torch.matmul(SU2xP_op, U_tau))
    
    def operate(self, target, phi, rho, tau):
        operator = self.op(phi, rho, tau)
        if tau%2 == 0:
            return torch.matmul(operator, target)
        else:
            return torch.matmul(operator, target.conj())