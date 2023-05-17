from typing import Union, List

class Irrep():
    def __init__(self, 
                 j: Union[int, float, str, tuple[Union[int, float]], List[Union[int, float]]], 
                 p: Union[int, str] = None, 
                 t: Union[int, str] = None,
                 j_tol: float = 1E-5) -> None:
        
        self._parity2num = {'e':1, 'o':-1}
        self._num2parity = {1:'e', -1:'o'}

        if p == None or t == None:
            if isinstance(j, str):
                string = j.strip()
                j = string[:-2].split('/')
                if len(j) != 1:
                    j = float(j[0])/float(j[1])
                else:
                    j = float(j[0])
                p = string[-2]
                t = string[-1]
            elif isinstance(j, tuple) or isinstance(j, List):
                j, p, t = j
            else:
                raise ValueError(f'If p and t are not given, they must be included in j, but given j = {j}.')

        if j % 1 <= j_tol:
            numer = int(j)
            denum = 1
        elif (2 * j) % 1 <= j_tol:
            numer = int(2 * j)
            denum = 2
        else:
            raise ValueError(f'Variable j must be integer or half-integer with tolerance = {j_tol}, but given j = {j}.')

        if p in self._parity2num:
            p = self._parity2num[p]
        elif p not in self._num2parity:
            raise ValueError(f'Variable p must be parity \'e\' or \'o\' or their respective value 1 or -1, but given p = {p}.')
        
        if t in self._parity2num:
            t = self._parity2num[t]
        elif t not in self._num2parity:
            raise ValueError(f'Variable t must be parity \'e\' or \'o\' or their respective value 1 or -1, but given t = {t}.')

        self._numer = numer
        self._denum = denum
        self._p = p
        self._t = t

    @property
    def numer(self) -> int:
        return self._numer

    @property
    def denum(self) -> int:
        return self._denum

    @property
    def j(self) -> float:
        return float(self.numer/self.denum)

    @property
    def p(self) -> int:
        return self._p

    @property
    def t(self) -> int:
        return self._t

    @property
    def dim(self) -> int:
        return int(2 * self.f + 1)
    
    def __repr__(self) -> str:
        string = ''
        if self.denum == 1:
            string += f'{self.numer}'
        else:
            string += f'{self.numer}/{self.denum}'
        return string + f'{self._num2parity[self.p]}{self._num2parity[self.t]}'