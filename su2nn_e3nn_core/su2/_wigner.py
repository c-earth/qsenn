import torch

def su2_generators(j):
    ms = torch.arange(- float(j), float(j), 1, dtype = torch.float64)
    ladder_factors = ((j - ms) * (j + ms + 1)) ** 0.5
    Jx = (torch.diag(ladder_factors, diagonal = 1) + torch.diag(ladder_factors, diagonal = -1)) * 1j / 2
    Jy = (torch.diag(ladder_factors, diagonal = 1) - torch.diag(ladder_factors, diagonal = -1)) / 2 * (1 + 0j)
    Jz = torch.diag(torch.arange(- j, j+1, 1, dtype = torch.float64)) * 1j
    return Jx, Jy, Jz

def wigner_D(j, alpha, beta, gamma):
    Jx, Jy, Jz = su2_generators(j)
    return torch.matrix_exp(-alpha * Jz) @ torch.matrix_exp(-beta * Jy) @ torch.matrix_exp(-gamma * Jz)

def clebsch_gordan(j1, j2, j3):
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = torch.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype=torch.float64)
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = clebsch_gordan_coeff(
                        (j1, m1), (j2, m2), (j3, m1 + m2)
                    )
    return mat


def clebsch_gordan_coeff(idx1, idx2, idx3):
    from fractions import Fraction
    from math import factorial

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v), f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3)
        )
    C = C * S
    return C