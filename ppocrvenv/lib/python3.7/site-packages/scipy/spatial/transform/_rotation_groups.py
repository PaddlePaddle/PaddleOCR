import numpy as np
from scipy.constants import golden as phi


def icosahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    a = 0.5
    b = 0.5 / phi
    c = phi / 2
    g2 = np.array([[+a, +b, +c, 0],
                   [+a, +b, -c, 0],
                   [+a, +c, 0, +b],
                   [+a, +c, 0, -b],
                   [+a, -b, +c, 0],
                   [+a, -b, -c, 0],
                   [+a, -c, 0, +b],
                   [+a, -c, 0, -b],
                   [+a, 0, +b, +c],
                   [+a, 0, +b, -c],
                   [+a, 0, -b, +c],
                   [+a, 0, -b, -c],
                   [+b, +a, 0, +c],
                   [+b, +a, 0, -c],
                   [+b, +c, +a, 0],
                   [+b, +c, -a, 0],
                   [+b, -a, 0, +c],
                   [+b, -a, 0, -c],
                   [+b, -c, +a, 0],
                   [+b, -c, -a, 0],
                   [+b, 0, +c, +a],
                   [+b, 0, +c, -a],
                   [+b, 0, -c, +a],
                   [+b, 0, -c, -a],
                   [+c, +a, +b, 0],
                   [+c, +a, -b, 0],
                   [+c, +b, 0, +a],
                   [+c, +b, 0, -a],
                   [+c, -a, +b, 0],
                   [+c, -a, -b, 0],
                   [+c, -b, 0, +a],
                   [+c, -b, 0, -a],
                   [+c, 0, +a, +b],
                   [+c, 0, +a, -b],
                   [+c, 0, -a, +b],
                   [+c, 0, -a, -b],
                   [0, +a, +c, +b],
                   [0, +a, +c, -b],
                   [0, +a, -c, +b],
                   [0, +a, -c, -b],
                   [0, +b, +a, +c],
                   [0, +b, +a, -c],
                   [0, +b, -a, +c],
                   [0, +b, -a, -c],
                   [0, +c, +b, +a],
                   [0, +c, +b, -a],
                   [0, +c, -b, +a],
                   [0, +c, -b, -a]])
    return cls.from_quat(np.concatenate((g1, g2)))


def octahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    c = np.sqrt(2) / 2
    g2 = np.array([[+c, 0, 0, +c],
                   [0, +c, 0, +c],
                   [0, 0, +c, +c],
                   [0, 0, -c, +c],
                   [0, -c, 0, +c],
                   [-c, 0, 0, +c],
                   [0, +c, +c, 0],
                   [0, -c, +c, 0],
                   [+c, 0, +c, 0],
                   [-c, 0, +c, 0],
                   [+c, +c, 0, 0],
                   [-c, +c, 0, 0]])
    return cls.from_quat(np.concatenate((g1, g2)))


def tetrahedral(cls):
    g1 = np.eye(4)
    c = 0.5
    g2 = np.array([[c, -c, -c, +c],
                   [c, -c, +c, +c],
                   [c, +c, -c, +c],
                   [c, +c, +c, +c],
                   [c, -c, -c, -c],
                   [c, -c, +c, -c],
                   [c, +c, -c, -c],
                   [c, +c, +c, -c]])
    return cls.from_quat(np.concatenate((g1, g2)))


def dicyclic(cls, n, axis=2):
    g1 = cyclic(cls, n, axis).as_rotvec()

    thetas = np.linspace(0, np.pi, n, endpoint=False)
    rv = np.pi * np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    g2 = np.roll(rv, axis, axis=1)
    return cls.from_rotvec(np.concatenate((g1, g2)))


def cyclic(cls, n, axis=2):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rv = np.vstack([thetas, np.zeros(n), np.zeros(n)]).T
    return cls.from_rotvec(np.roll(rv, axis, axis=1))


def create_group(cls, group, axis='Z'):
    if not isinstance(group, str):
        raise ValueError("`group` argument must be a string")

    permitted_axes = ['x', 'y', 'z', 'X', 'Y', 'Z']
    if axis not in permitted_axes:
        raise ValueError("`axis` must be one of " + ", ".join(permitted_axes))

    if group in ['I', 'O', 'T']:
        symbol = group
        order = 1
    elif group[:1] in ['C', 'D'] and group[1:].isdigit():
        symbol = group[:1]
        order = int(group[1:])
    else:
        raise ValueError("`group` must be one of 'I', 'O', 'T', 'Dn', 'Cn'")

    if order < 1:
        raise ValueError("Group order must be positive")

    axis = 'xyz'.index(axis.lower())
    if symbol == 'I':
        return icosahedral(cls)
    elif symbol == 'O':
        return octahedral(cls)
    elif symbol == 'T':
        return tetrahedral(cls)
    elif symbol == 'D':
        return dicyclic(cls, order, axis=axis)
    elif symbol == 'C':
        return cyclic(cls, order, axis=axis)
    else:
        assert False
