# Memo to Achyuthan: what the checker must catch

From John, based on the seat-152 baseline (416 attempts) plus the seat-154 partial run. Full counts in
`01-baseline-and-taxonomy.md`. Ordered by how many attempts each one cost.

Two facts to design around:

1. **The model resends identical code after most messages**: level 4 60/60 repair rounds, level 3 52/64. A
   better message only helps if it changes what the model does; check in the A/B logs whether "same code"
   drops, not only the score.
2. **Two of the existing `enrich()` hints in `agent.py` actively mislead.** Fixing those may matter as much
   as adding locate/state.

---

### 1. Level 4: whole tensor in one tile (70 attempts, every run, 60/60 resent unchanged)

- **Model sees:** `1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128` + "loop over the partition dimension in chunks of at most 128..."
- **Really at fault:** `sbuf_stationary = nl.ndarray((K, M), ..., buffer=nl.sbuf)` (and the same for `rhs`): the tile's size is taken from the input's shape, so it grows with K.
- **Useful message, as a rule:** at that line `sbuf_stationary` is (256, 256) in sbuf because it was sized from `lhsT`; a tile's row count must stay <= 128 for every input, so it cannot be sized from an input's shape. Also say **which shapes passed and which failed**: it passed only where K <= 128.
- **Covered by:** `state`, plant "whole tensor in one tile". Missing: the passed/failed shape list, which comes from `agent._grade` reporting only the first failure.

### 2. Level 3: matmul result given one dimension (62/84 attempts contain it, 19 fail on it directly)

- **Model sees:** `SBUF and PSUM tensors must have at least 2 dimensions` + "give a length-N vector the shape (1, N) or (N, 1)".
- **Really at fault:** `out = nl.ndarray(shape=lhsT.shape[1:], ...)` gives `(64,)`, and `psum = nl.ndarray(shape=out.shape, ...)` inherits it.
- **Useful message, as a rule:** the result of `nc_matmul` takes its first dimension from stationary's second and its second from moving's second; this tile has 1 dimension because its shape came from one input only.
- **The current hint is harmful:** it produced type 3d (16 attempts, operands shrunk to (1, M) / (K, 1)) and, on seat-154, the `cannot reshape ... into shape (1,64)` loop (16 attempts, 0 contain `reshape`). **Remove the "(1, N) or (N, 1)" advice** from the `must have at least 2 dimensions` branch of `enrich()`.
- **Covered by:** the existing plant uses a 1-D *operand*. Add a plant with `out = nl.ndarray(shape=lhsT.shape[1:], ...)` and `psum` sized from `out`, and check the message names the result-shape rule.

### 3. Level 3: one scratch tile reused for operands of different shapes (49 attempts)

- **Model sees:** `dma_copy requires src and dst to have the same number of elements, got src=8192, dst=65536` + "If you want a 128x512 piece of a bigger tensor, write t = nl.ndarray((128, 512)..."
- **Really at fault:** `nisa.dma_copy(dst=sbuf_tile, src=lhsT)`: `sbuf_tile` is (128, 512), `lhsT` is (128, 64). The same tile is then used for `rhs` and for the result.
- **Useful message, as a rule:** at that line dst is (128, 512) in sbuf and src is (128, 64) in HBM; each operand needs a tile of its own, shaped like what is copied into it.
- **The current hint's example is the bug:** it suggests a (128, 512) tile, which is exactly the tile the model allocated. Make the example use the shapes from the failing line, or drop it.
- **Covered by:** `state`, plant "tile too small for the copy".

### 4. Level 1: right function, wrong module (60 attempts) - quick fix in `agent.available_names`

- **Model sees:** `` module 'nki.isa' has no attribute 'multiply' `nki.isa` has no `multiply`, and nothing similar exists. Its real names include: ... ``
- **Really at fault:** `nisa.tensor_scalar(dst=tile, data=0.5, op0=nisa.multiply)`: `multiply` lives in `nki.language`.
- **Useful message, as a rule:** `multiply` is not in `nki.isa`; it is in `nki.language`. When a name is missing from one NKI module, look it up in the other two before saying it does not exist.
- **Not a locate/state problem:** `available_names()` only searches the module in the error. This is a small, safe change.

### 5. Level 1 and 4: result copied out through an operand's tile (10 attempts on level 4, also in level 3)

- **Model sees:** `value array of shape (65536,) could not be broadcast to indexing result of shape (16384,)`.
- **Really at fault:** `nisa.tensor_copy(dst=sbuf_lhsT, src=psum)`: the (M, N) result into the (K, M) operand tile.
- **Useful message, as a rule:** at that line dst is (128, 128) in sbuf and src is (128, 512) in psum; the destination of a copy must have the source's shape.
- **Covered by:** `state` (`tensor_copy` element-count rule). Worth a plant, because it fails on the single-tile shape too.

---

### What A and B cannot fix (for the write-up and for Perumal)

Level 1 fails on the **algorithm** in 160/160 attempts: the average is computed as `nc_matmul(tile, tile)`
times `0.5`. No message fires on that because no attempt reaches the numerical check. Even if every API error
were fixed, level 1 would stay below 1.0. The likely cause is the prompt: the API card's only scaling example is
`tensor_scalar(..., operand0=0.5)` and it has no reduction example. That is a prompt change (Perumal's
territory), and it must be measured separately from A/B.
