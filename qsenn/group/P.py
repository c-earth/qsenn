import torch

from _G import MulIr

p2n = {'o': -1, 'e': 1}
n2p = {-1: 'o', 1: 'e'}

class IrrepP():
    def __init__(self, p):
        if isinstance(p, IrrepP):
            p = p.p
        elif p in p2n:
            p = p2n[p]

        if p not in n2p:
            raise ValueError(f'parity must be an integer -1, +1, or a string \'o\', \'e\', but got {p}')
        else:
            self._p = p

    @property
    def p(self):
        return self._p
    
    @property
    def dim(self):
        return 1
    
    def __repr__(self):
        return f'{n2p[self.p]}'
    
    def __mul__(self, other):
        other = IrrepP(other)
        p = self.p * other.p
        yield IrrepP(p)

    def __rmul__(self, other):
        if not isinstance(other, int):
            raise ValueError(f'multiplicity must be an integer, but got {type(other)}')
        elif other <= 0:
            raise ValueError(f'multiplicity must be positive, but got {other}')
        return IrrepsP([(other, self)])
    
    def __add__(self, other):
        return IrrepsP(self) + IrrepsP(other)
    
    def is_scalar(self):
        return self.p == 1
    
    def D(self, rho):
        if not isinstance(rho, int):
            raise ValueError(f'parity parameterization must be an integer, but got {type(rho)}')
        elif rho < 0:
            raise ValueError(f'parity parameterization must be non-negative, but got {rho}')
        return torch.tensor([[self.p ** rho]])
    
    @classmethod
    def iterator(cls):
        for p in n2p:
            yield IrrepP(p)

class MulIrrepP(MulIr):
    def __new__(cls, mul, ir):
        if not isinstance(mul, int):
            raise ValueError(f'multiplicity must be an integer, but got {type(mul)}')
        elif mul <= 0:
            raise ValueError(f'multiplicity must be positive, but got {mul}')
        
        if not isinstance(ir, IrrepP):
            raise ValueError(f'expected IrrepP object, but got {type(ir)}')
        
        return super().__new__(cls, (mul, ir))

class IrrepsP(tuple):
    def __new__(cls, irreps):
        if isinstance(irreps, IrrepsP):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, IrrepP):
            out.append(MulIrrepP(1, IrrepP(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(MulIrrepP(mul, ir))
        return super().__new__(cls, out)

    def slices(self):
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(self, *size, normalization="component", requires_grad=False, dtype=None, device=None):
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]

        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir):
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def __add__(self, irreps):
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other):
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        return Irreps(super().__rmul__(other))

    def simplify(self) -> 'Irreps':
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self):
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return IrrepsP(out)

    def sort(self):
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self):
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self):
        return [l for mul, (l, p) in self for _ in range(mul)]

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D(self, rho):
        return direct_sum(*[ir.D(rho) for mul, ir in self for _ in range(mul)])