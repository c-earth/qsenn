from qsenn.group.IrrepTP import IrrepTP
from qsenn.group.SU2 import IrrepSU2
from qsenn.group.P import IrrepP

class IrrepSU2xP(IrrepTP):
    @property
    def parents(self):
        return (IrrepSU2, IrrepP)