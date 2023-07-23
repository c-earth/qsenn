from qsenn.group.IrrepTP import IrrepTP
from qsenn.group.SU2 import IrrepSU2
from qsenn.group.P import IrrepP

class IrrepSU2xP(IrrepTP):
    @property
    def parents(self):
        return (IrrepSU2, IrrepP)
    
    @property
    def j(self):
        return self._2j/2

    @property
    def _2j(self):
        return self.labels[0]
    
    @property
    def p(self):
        return self.labels[1]