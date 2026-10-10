"""Imitation of nki.isa. See the package docstring: plumbing tests only."""
import numpy as np


def dma_copy(dst=None, src=None, **kw):
    if dst.arr.size != src.arr.size:
        raise ValueError(f"dma_copy requires src and dst to have the same number of elements, "
                         f"got src={src.arr.size}, dst={dst.arr.size}")
    for name, t in (("dst", dst), ("src", src)):
        if getattr(t.buffer, "name", "") == "sbuf" and t.shape[0] > 128:
            raise ValueError(f"dma_copy {name} partition dimension {t.shape[0]} exceeds maximum 128")
    dst.arr[...] = src.arr.reshape(dst.arr.shape)


def nc_matmul(dst=None, stationary=None, moving=None, **kw):
    out = stationary.arr.astype(np.float32).T @ moving.arr.astype(np.float32)
    dst.arr[...] += out.reshape(dst.arr.shape)


def tensor_copy(dst=None, src=None, **kw):
    dst.arr[...] = src.arr.reshape(dst.arr.shape)


def tensor_scalar(dst=None, data=None, op0=None, operand0=None, **kw):
    dst.arr[...] = op0(data.arr, operand0)
