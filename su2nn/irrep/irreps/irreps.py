from irrep import Irrep
from mulirrep import MulIrrep

class Irreps():
    def __init__(self, irreps):
        if isinstance(irreps, Irreps):
            return irreps
        
        if isinstance(irreps, MulIrrep):
            irreps = [irreps]

        elif isinstance(irreps, Irrep):
            irreps = [MulIrrep(1, irreps)]

        elif isinstance(irreps, str):
            irreps = irreps.split('+')
            irreps = [MulIrrep(mulirrep) for mulirrep in irreps]
        
        elif isinstance(irreps, tuple) or isinstance(irreps, list):
            irreps = [MulIrrep(mulirrep) for mulirrep in irreps]

        self._irreps = irreps