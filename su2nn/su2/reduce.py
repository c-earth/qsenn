import torch

from math import factorial
from fractions import Fraction


def clebsch_gordan(j_in1, j_in2, j_out):
    assert isinstance(j_in1, (int, float, Fraction)) and (2 * j_in1) % 1 == 0
    assert isinstance(j_in2, (int, float, Fraction)) and (2 * j_in2) % 1 == 0
    assert isinstance(j_out, (int, float, Fraction)) and (2 * j_out) % 1 == 0
    CG = torch.zeros((int(2 * j_in1 + 1), int(2 * j_in2 + 1), int(2 * j_out + 1)), dtype = torch.float64)
    if abs(j_in1 - j_in2) <= j_out <= (j_in1 + j_in2):
        for m_in1 in [x / 2 for x in range(-int(2 * j_in1), int(2 * j_in1) + 1, 2)]:
            for m_in2 in [x / 2 for x in range(-int(2 * j_in2), int(2 * j_in2) + 1, 2)]:
                if abs(m_in1 + m_in2) <= j_out:
                    CG[int(j_in1 + m_in1), int(j_in2 + m_in2), int(j_out + m_in1 + m_in2)] = clebsch_gordan_coeff(j_in1, m_in1, j_in2, m_in2, j_out, m_in1 + m_in2)
    return CG


def clebsch_gordan_coeff(j_in1, m_in1, j_in2, m_in2, j_out, m_out):
    
    def f(n):
        return factorial(round(n))
    
    if m_out != m_in1 + m_in2:
        return 0
    
    cgc  = 2 * j_out + 1
    cgc *= f(j_out + j_in1 - j_in2)
    cgc *= f(j_out - j_in1 + j_in2)
    cgc *= f(j_in1 + j_in2 - j_out)
    cgc /= f(j_in1 + j_in2 + j_out + 1)
    cgc *= f(j_out + m_out)
    cgc *= f(j_out - m_out)
    cgc *= f(j_in1 + m_in1)
    cgc *= f(j_in1 - m_in1)
    cgc *= f(j_in2 + m_in2)
    cgc *= f(j_in2 - m_in2)
    cgc = cgc ** 0.5

    add = 0
    k = 0
    while True:
        try:
            tmp  = -1 ** k / f(k)
            tmp /= f(j_in1 + j_in2 - j_out - k)
            tmp /= f(j_in1 - m_in1 - k)
            tmp /= f(j_in2 + m_in2 - k)
            tmp /= f(j_out - j_in2 + m_in1 + k)
            tmp /= f(j_out - j_in1 - m_in2 + k)
            add += tmp
            k += 1
        except:
            break
    return cgc * add

def clebsch_gordan_coeff_old(idx1, idx2, idx3):
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

if __name__ == '__main__':
    idx1 = (1, 1)
    idx2 = (1, 1)
    idx3 = (2, 2)
    import time
    start = time.time()
    print(clebsch_gordan_coeff(*idx1, *idx2, *idx3))
    print(time.time() - start)
    start = time.time()
    print(clebsch_gordan_coeff_old(idx1, idx2, idx3))
    print(time.time() - start)