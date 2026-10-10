"""Imitation of nki.isa. See the package docstring: plumbing tests only."""
import numpy as np


def dma_copy(dst=None, src=None, **kw):
    if dst.arr.size != src.arr.size:
        raise AssertionError(f"dma_copy requires src and dst to have the same number of elements, "
                         f"got src={src.arr.size}, dst={dst.arr.size}")
    for name, t in (("dst", dst), ("src", src)):
        if getattr(t.buffer, "name", "") == "sbuf" and t.shape[0] > 128:
            raise AssertionError(f"dma_copy {name} partition dimension {t.shape[0]} exceeds maximum 128")
    dst.arr[...] = src.arr.reshape(dst.arr.shape)


def nc_matmul(dst=None, stationary=None, moving=None, **kw):
    if stationary.shape[0] > 128:
        raise AssertionError(f"Matmul contraction dimension {stationary.shape[0]} exceeds pmax=128")
    if stationary.shape[1] > 128:
        raise AssertionError(f"Matmul stationary free dimension {stationary.shape[1]} exceeds "
                             f"gemm_stationary_fmax=128")
    if moving.shape[1] > 512:
        raise AssertionError(f"Matmul moving free dimension {moving.shape[1]} exceeds "
                             f"gemm_moving_fmax=512")
    if dst.shape[-1] > 512:
        # Seen on the real simulator with both operands in range: it names the moving operand.
        raise AssertionError(f"Matmul moving free dimension {dst.shape[-1]} exceeds max 512 for "
                             f"nc_version=nc_version.gen3")
    out = stationary.arr.astype(np.float32).T @ moving.arr.astype(np.float32)
    dst.arr[...] += out.reshape(dst.arr.shape)


def tensor_copy(dst=None, src=None, **kw):
    dst.arr[...] = src.arr.reshape(dst.arr.shape)


def tensor_scalar(dst=None, data=None, op0=None, operand0=None, **kw):
    dst.arr[...] = op0(data.arr, operand0)
