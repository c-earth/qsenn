import abc

from qsenn.tensor.direct_sum import direct_sum

class IrrepInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass
    
    @abc.abstractmethod
    def __mul__(self):
        pass
    
    @abc.abstractmethod
    def D(self):
        pass

    @classmethod
    @abc.abstractmethod
    def iterator(cls):
        pass

class Irrep(IrrepInterface):
    def __rmul__(self, mul):
        return self.Irreps([(mul, self)])

    def __add__(self, other):
        return self.Irreps(self) + self.Irreps(other)
    
    @classmethod
    def MulIrrep(cls, mulirrep):
        if isinstance(mulirrep, MulIrrep):
            return mulirrep
        
        if isinstance(mulirrep, cls):
            mul = 1
            irrep = mulirrep
        elif isinstance(mulirrep, str):
            if 'x' in mulirrep:
                mul, irrep = mulirrep.split('x')
            else:
                mul = 1
                irrep = mulirrep
        elif isinstance(mulirrep, tuple):
            if isinstance(mulirrep[0], int) and isinstance(mulirrep[1], (cls, tuple, str)):
                mul, irrep = mulirrep
            else:
                mul = 1
                irrep = mulirrep
        else:
            raise ValueError(f'unable to interpret {mulirrep} as MulIrrep of this Irrep')
        
        try:
            mul = int(mul)
            if mul < 0:
                raise ValueError(f'multiplicity must be non-negative, but got {mul}')
            irrep = cls(irrep)
        except:
            raise ValueError(f'unable to interpret {mulirrep} as MulIrrep of this Irrep')
        
        return MulIrrep((mul, irrep))

    @classmethod
    def Irreps(cls, irreps):
        if isinstance(irreps, Irreps):
            return irreps
        
        out = []
        try:
            if isinstance(irreps, MulIrrep):
                out.append(irreps)
            elif isinstance(irreps, cls):
                out.append(MulIrrep((1, irreps)))
            elif isinstance(irreps, str):
                if '+' in irreps:
                    for mulirrep in irreps.split('+'):
                        out.append(cls.MulIrrep(mulirrep))
                else:
                    out.append(cls.MulIrrep(irreps))
            elif isinstance(irreps[0], int):
                out.append(cls.MulIrrep(irreps))
            else:
                for mulirrep in irreps:
                    out.append(cls.MulIrrep(mulirrep))
        except:
            raise ValueError(f'unable to interpret {irreps} as Irreps of this Irrep')
        
        return Irreps(out)

class MulIrrep(tuple):
    def __new__(cls, mulirrep):
        return super().__new__(cls, mulirrep)

    @property
    def mul(self):
        return self[0]
    
    @property
    def irrep(self):
        return self[1]
    
    @property
    def dim(self):
        return self.mul * self.irrep.dim
    
    @property
    def Irrep(self):
        return type(self.irrep)

    def __repr__(self):
        return f'{self.mul}x{self.irrep}'

class Irreps(tuple):
    def __new__(cls, irreps):
        return super().__new__(cls, irreps)

    @property
    def dim(self):
        return sum(mulirrep.dim for mulirrep in self)

    @property
    def num(self):
        return sum(mulirrep.mul for mulirrep in self)
    
    @property
    def Irrep(self):
        if self.num == 0:
            raise ValueError('Irreps contains no Irrep')
        else:
            return self[0].Irrep

    def __repr__(self):
        return '+'.join(f'{mulirrep}' for mulirrep in self)
    
    def __getitem__(self, i):
        irreps = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(irreps)
        return irreps

    def __contains__(self, irrep):
        irrep = self.Irrep(irrep)
        return irrep in [mulirrep.irrep for mulirrep in self]
    
    def __add__(self, irreps):
        if not isinstance(irreps, Irreps):
            raise ValueError(f'direct sum must be between Irrepses, but get {type(irreps)}')
        elif self.Irrep != irreps.Irrep:
            raise ValueError(f'direct sum must be between Irrepses with the same Irrep, but get {self.Irrep}, and {irreps.Irrep}')
        else:
            return Irreps(super().__add__(irreps))

    def __mul__(self, other):
        if isinstance(other, Irreps):
            raise NotImplementedError
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        return Irreps(super().__rmul__(other))

    def count(self, irrep):
        irrep = self.Irrep(irrep)
        return sum(mulirrep.mul for mulirrep in self if mulirrep.irrep == irrep)
    
    def remove_zero_muls(self):
        out = [mulirrep for mulirrep in self if mulirrep.mul > 0]
        return Irreps(out)

    def simplify(self):
        out = []
        for mulirrep in self:
            if len(out) != 0 and out[-1].irrep == mulirrep.irrep:
                out[-1] = MulIrrep((out[-1].mul + mulirrep.mul, mulirrep.irrep))
            elif mulirrep.mul > 0:
                out.append(mulirrep)
        return Irreps(out)

    def D(self, *args):
        return direct_sum(*[mulirrep.irrep.D(*args) for mulirrep in self for _ in range(mulirrep.mul)])