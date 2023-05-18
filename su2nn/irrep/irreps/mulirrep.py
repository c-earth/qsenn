from irrep import Irrep

class MulIrrep():
    def __init__(self, mul, irrep = None):
        if isinstance(mul, MulIrrep):
            return mul
        
        if isinstance(mul, Irrep):
            irrep = mul
            mul = 1

        elif irrep == None:
            if isinstance(mul, str) and 'x' in mul:
                mul = mul.strip()
                mul, irrep = mul.split('x')
                
            elif isinstance(mul, tuple) or isinstance(mul, list):
                if len(mul) == 2:
                    mul, irrep = mul
                else:
                    raise ValueError(f'_MulIrrep take only 2 arguments, but {len(mul)} are given.')
                
            else:
                raise ValueError(f'If irrep is not given, they must be included in mul, but mul = {mul} is given.')

            try:
                mul = eval(mul)
            except:
                raise ValueError(f'Cannot Evaluate mul = {mul} to numerical value.')
            
            if mul < 0:
                raise ValueError(f'mul must be non-negative, but mul = {mul} is given.')
            
            if not isinstance(mul, int):
                raise ValueError(f'Variable j must be integer, but mul = {mul} is given.')

        irrep = Irrep(irrep)

        self._mul = mul
        self._irrep = irrep

    @property
    def mul(self):
        return self._mul

    @property
    def irrep(self):
        return self._irrep
    
    def __repr__(self):
        return f'{self.mul}x{self.irrep}'