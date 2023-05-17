class Irrep():
    def __init__(self, j, p = None, t = None, j_tol = 1E-5):
        self._par2num = {'e':1, 'o':-1}
        self._num2par = {1:'e', -1:'o'}

        if isinstance(j, Irrep):
            return j

        if p == None or t == None:
            if isinstance(j, str) and len(j) >= 3:
                string = j.strip()
                j = string[:-2]
                p = string[-2]
                t = string[-1]

                try:
                    j = float(eval(j))
                except:
                    raise ValueError(f'Cannot Evaluate j = {j} to float.')
                
            elif isinstance(j, tuple) or isinstance(j, list):
                if len(j) == 3:
                    j, p, t = j
                else:
                    raise ValueError(f'Irrep take only 3 arguments, but {len(j)} are given.')
                
            else:
                raise ValueError(f'If p or t is not given, they must be included in j, but j = {j} is given.')

        if j < 0:
            raise ValueError(f'j must be non-negative, but j = {j} is given.')
        
        if min((1 - (j % 1)), (j % 1)) <= j_tol:
            numer = int(j)
            denum = 1
        elif min((1 - ((2 * j) % 1)), ((2 * j) % 1)) <= j_tol:
            numer = int(2 * j)
            denum = 2
        else:
            raise ValueError(f'j must be (half-)integer with tolerance = {j_tol}, but j = {j} is given.')

        if p in self._par2num:
            p = self._par2num[p]
        elif p not in self._num2par:
            raise ValueError(f'p must be parity \'e\' or \'o\' or their respective value 1 or -1, but given p = {p}.')
        
        if t in self._par2num:
            t = self._par2num[t]
        elif t not in self._num2par:
            raise ValueError(f't must be parity \'e\' or \'o\' or their respective value 1 or -1, but given t = {t}.')

        self._numer = numer
        self._denum = denum
        self._p = p
        self._t = t

    @property
    def numer(self):
        return self._numer

    @property
    def denum(self):
        return self._denum

    @property
    def j(self):
        return float(self.numer/self.denum)

    @property
    def p(self):
        return self._p

    @property
    def t(self):
        return self._t

    @property
    def dim(self):
        return int(2 * self.f + 1)
    
    def __repr__(self):
        if self.denum == 1:
            j_str = f'{self.numer}'
        else:
            j_str = f'{self.numer}/{self.denum}'
        return f'{j_str}{self._num2parity[self.p]}{self._num2parity[self.t]}'
    
class _MultiIrrep():
    def __init__(self, mul, irrep = None, mul_tol = 1E-5):
        if isinstance(mul, _MultiIrrep):
            return mul

        if irrep == None:
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
    
class Irreps():
    def __init__(self):
        pass