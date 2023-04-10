r'''Allows for clean lookup of Irreducible representations of :math:`SU(2)`

Examples
--------
Create a scalar representation (:math:`l=0`) of even parity and odd time reversal symmetry.

>>> from su2nn.su2 import irrep
>>> irrep.l0e == Irrep("0e")
True

>>> from su2nn.su2.irrep import l1o, l2o
>>> l1o + l2o == Irrep('1o') + Irrep('2o')
True
'''

from .._irreps import Irrep


def __getattr__(name: str):
    r'''Creates an Irreps obeject by reflection

    Parameters
    ----------
    name : string
        the su2 object name prefixed by l. Example: l1oe == Irrep('1oe')

    Returns
    -------
    `e3nn.o3.Irrep`
        irreducible representation of :math:`SU(2)`
    '''

    prefix, *ir = name
    if prefix != 'l' or not ir:
        raise AttributeError(f'\'su2nn.su2.irrep\' module has no attribute \'{name}\'')

    try:
        return Irrep(''.join(ir))
    except (ValueError, AssertionError):
        raise AttributeError(f'\'su2nn.su2.irrep\' module has no attribute \'{name}\'')