class Irrep():
    def __init__(self):
        pass

    @property
    def dim(self):
        pass
    
    def __repr__(self):
        pass
    
    def __mul__(self, other):
        pass

    def __rmul__(self, mul):
        pass

    def __add__(self, other):
        return Irreps(self) + Irreps(other)
    
    def D(self):
        pass

    @classmethod
    def iterator(cls):
        pass

class MulIr(tuple):
    def __new__(cls):
        pass

    @property
    def mul(self):
        return self[0]
    
    @property
    def ir(self):
        return self[1]
    
    @property
    def dim(self):
        return self.mul * self.ir.dim

    def __repr__(self):
        return f'{self.mul}x{self.ir}'

class Irreps(tuple):
    def __new__(cls):
        pass