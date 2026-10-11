"""Imitation of nki.language. See the package docstring: plumbing tests only."""
import numpy as np


class _Region:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


sbuf, psum, shared_hbm = _Region("sbuf"), _Region("psum"), _Region("shared_hbm")
float32 = np.float32
bfloat16 = np.float32
multiply = np.multiply


class tile_size:
    pmax = 128
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512


class _Tensor:
    def __init__(self, arr, buffer):
        self.arr = arr
        self.buffer = buffer

    shape = property(lambda self: self.arr.shape)
    dtype = property(lambda self: self.arr.dtype)

    def __getitem__(self, idx):
        return _Tensor(self.arr[idx], self.buffer)

    def ap(self, pattern):
        """A strided view, [[step, count], ...] over this tile's storage. The first pair must walk
        the partitions; the message is the real simulator's, copied from a log."""
        part = int(np.prod(self.arr.shape[1:]))
        if pattern[0][0] != part:
            raise AssertionError(
                f"ap() pattern has invalid partition stride. Partition step {pattern[0][0]} must "
                f"equal tensor free dimension size {part}. Pattern: {pattern}, tensor shape: "
                f"{tuple(self.arr.shape)}")
        flat = np.ascontiguousarray(self.arr).reshape(-1)
        item = flat.itemsize
        view = np.lib.stride_tricks.as_strided(
            flat, shape=[c for _, c in pattern], strides=[st * item for st, _ in pattern])
        return _Tensor(view, self.buffer)


def affine_range(*args):
    return range(*args)


def ndarray(shape, dtype=np.float32, buffer=None):
    shape = tuple(shape)
    if buffer in (sbuf, psum) and len(shape) < 2:
        raise AssertionError("SBUF and PSUM tensors must have at least 2 dimensions "
                             "(partition-dim and free-dim)")
    return _Tensor(np.zeros(shape, dtype), buffer)


def sum(view, axis, dst=None, **kw):
    out = view.arr.sum(axis=tuple(axis))
    if dst is not None:
        dst.arr[...] = out
        return dst
    return _Tensor(out, sbuf)
