import os
import jax.numpy
from jax.numpy import sqrt


def max(a, b=None):
    if b is None:
        values = list(a)
        result = values[0]
        for value in values[1:]:
            result = jax.numpy.maximum(result, value)
        return result

    return jax.numpy.maximum(a, b)


def min(a, b=None):
    if b is None:
        values = list(a)
        result = values[0]
        for value in values[1:]:
            result = jax.numpy.minimum(result, value)
        return result

    return jax.numpy.minimum(a, b)


from jax.numpy import power as pow
from jax.numpy import pi
from itertools import product
import jax.numpy as np

assert sqrt
assert pow
assert pi
assert product

# Set MADJAX_PRECISION=float64 for double precision, default is float32
_precision = os.environ.get("MADJAX_PRECISION", "float32").lower()
_float_dtype = np.float64 if _precision == "float64" else np.float32
_complex_dtype = np.complex128 if _precision == "float64" else np.complex64

import numpy as _numpy
_1j = _numpy.asarray(1j, dtype=_numpy.complex128 if _precision == "float64" else _numpy.complex64)


def complex(*v):
    if len(v) == 1:
        return np.asarray(v, dtype=_complex_dtype)
    else:
        return np.asarray(v[0] + _1j * v[1], dtype=_complex_dtype)
