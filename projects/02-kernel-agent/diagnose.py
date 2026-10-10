#!/usr/bin/env python3
"""
diagnose.py -- say WHERE in the kernel a failure happened, not only what the simulator said.

The harness reports a crash as, for example,

    raised ValueError: cannot reshape array of size 32768 into shape (1,64)

That sentence comes from inside the simulator. It describes the simulator's symptom, and whoever
wrote the kernel has to guess which of their own lines caused it. A kernel that never calls
reshape can still produce it, and "do not reshape" is then an instruction about code that is not
there. The missing half is cheap and exact: the traceback already knows which line of the
candidate was executing. This module reads it out, so the feedback can say

    line 23, `nisa.nc_matmul(dst=res, stationary=lhs_tile, moving=rhs_tile)`, raised ValueError: ...

It names a line of the model's OWN code. It never says what the line should be.

The second half is what was true on that line when it failed. The kernel's frame is still in the
traceback with every variable in it, so the shapes are not guesses:

    At that line: `result_psum` is (1, 64) in psum; `lhs_tile` is (128, 64) in sbuf; `rhs_tile`
    is (128, 512) in sbuf. nc_matmul's result has stationary's second dimension as its first ...

Shapes and values are the kernel's own, read at the moment of failure. A rule is added only when
those shapes break it, and it is stated as the rule -- which dimension must match which -- never
as the shape to type in. Nothing here can reject a kernel: it speaks only after the simulator
has already raised.

    python diagnose.py --selftest      # plant known bugs in the reference kernels, check both
    python diagnose.py --replay LOG    # what the model WOULD have been told, for every failing
                                       # kernel in an attempt log. Calls no model.

WHAT IS VERIFIED AND WHAT IS NOT. --selftest needs `nki`, so it proves something only where the
Neuron SDK is installed: run it in the seat pod before trusting a located line. devtools/fake_nki
is a stand-in for exercising the plumbing on a laptop and says nothing about the real simulator.
"""

import argparse
import ast
import math
import os
import re
import sys
import types

# Marks the located text inside a feedback string, so it can be shown to the model and also
# removed again. See shown() and key().
MARK = "\x00"

# where() and state() run inside the grader's error handler, in the middle of a measured run that
# takes the better part of an hour. A bug in them must cost one sentence of feedback, never the
# run. So they swallow their own failures and note them here, where --replay reports them.
ERRORS = []


def statement_at(source, lineno, limit=200):
    """The statement on `lineno`, on one line. A call split across several lines comes back whole;
    for a loop or an `if`, only its header line, since its body is not what failed."""
    lines = source.splitlines()
    plain = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.stmt):
                end = node.end_lineno or node.lineno
                if node.lineno <= lineno <= end and (best is None
                                                    or end - node.lineno <= best[0]):
                    best = (end - node.lineno, node)
    except SyntaxError:
        pass
    text = plain
    if best is not None and not hasattr(best[1], "body"):
        text = ast.get_source_segment(source, best[1]) or plain
    text = " ".join(text.split())
    return text[:limit] + (" ..." if len(text) > limit else "")


def _function_spans(source):
    try:
        return {n.name: (n.lineno, n.end_lineno or n.lineno) for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return {}


def _hit(exc, path, source=""):
    """(line, frame) of the candidate file that was executing when `exc` was raised, or None.

    Takes the INNERMOST frame that belongs to the candidate: if the kernel calls a helper of its
    own, the helper's line is the specific one. Walks the exception chain as well, because a
    simulator that catches an error and raises its own from above the kernel leaves the kernel's
    frame only on the original.
    """
    real = os.path.realpath(path)
    chain, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    spans = None
    for e in reversed(chain):
        hit, tb = None, e.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_filename == path or os.path.realpath(code.co_filename) == real:
                hit = (tb.tb_lineno, tb.tb_frame)
            elif source:
                # A decorator may recompile the function under another file name while keeping
                # its line numbers. Accept that only when the line really is inside a function
                # of that name in the candidate, so a simulator frame can never be mistaken for it.
                if spans is None:
                    spans = _function_spans(source)
                lo, hi = spans.get(code.co_name, (0, -1))
                if lo <= tb.tb_lineno <= hi and not os.path.exists(code.co_filename):
                    hit = (tb.tb_lineno, tb.tb_frame)
            tb = tb.tb_next
        if hit:
            return hit
    return None


def locate(exc, path, source=""):
    """The line of the candidate file that was executing when `exc` was raised, or None."""
    hit = _hit(exc, path, source)
    return hit[0] if hit else None


def where(exc, path, source):
    """'line N, `statement`, ' wrapped in MARK, or '' when the line cannot be found. Empty rather
    than a guess: a wrong line is worse than no line."""
    try:
        n = locate(exc, path, source)
        if not n:
            return ""
        return f"{MARK}line {n}, `{statement_at(source, n)}`, {MARK}"
    except Exception as e:
        ERRORS.append(f"where: {type(e).__name__}: {e}")
        return ""


def shown(text):
    """The feedback as the model sees it."""
    return text.replace(MARK, "")


def key(text):
    """The feedback with the located part removed: what an unmodified run would have said."""
    return re.sub(f"{MARK}.*?{MARK}", "", text, flags=re.S)


# ---------------------------------------------------------------- what was true on that line

REGIONS = ("sbuf", "psum", "shared_hbm", "private_hbm", "hbm")

# Positional order of the operands, for the calls whose rules are worth stating.
OPERANDS = {"dma_copy": ("dst", "src"), "tensor_copy": ("dst", "src"),
            "nc_matmul": ("dst", "stationary", "moving"), "ndarray": ("shape", "dtype", "buffer")}


def _region_name(buf):
    text = f"{getattr(buf, 'name', '')} {buf!r}".lower()
    return next((r for r in REGIONS if r in text), None) if buf is not None else None


class watching:
    """For the length of one simulation, remember which buffer each tile was allocated in.

    A tile's shape can be read off the tile. Where it lives is known for certain only at the
    moment it is allocated, from the buffer= it was given, so note it there. Same technique as
    the harness counting bytes by wrapping nisa.dma_copy, and the same caveat: it assumes the
    kernel calls nl.ndarray through the module. If it does not, nothing is recorded and the
    report simply leaves the buffer out.
    """

    def __init__(self):
        self.buffers, self._keep, self._nl = {}, [], None

    def __enter__(self):
        try:
            import nki.language as nl
        except ImportError:
            return self
        self._nl, self._real = nl, nl.ndarray
        known = {id(getattr(nl, r)): r for r in REGIONS if hasattr(nl, r)}
        real = self._real

        def ndarray(*args, **kw):
            out = real(*args, **kw)
            try:
                buf = kw.get("buffer", args[2] if len(args) > 2 else None)
                name = known.get(id(buf)) or _region_name(buf)
                if name:
                    self.buffers[id(out)] = name
                    self._keep.append(out)      # so the id cannot be reused while we look
            except Exception:
                pass
            return out

        nl.ndarray = ndarray
        return self

    def __exit__(self, *exc):
        if self._nl is not None:
            self._nl.ndarray = self._real
        return False


def _shape(val):
    try:
        shape = tuple(int(d) for d in val.shape)
    except Exception:
        return None
    return shape if not isinstance(val, type) else None


def _buffer(val, name, watch, params):
    for attr in ("buffer", "_buffer", "region", "memory"):
        try:
            found = _region_name(getattr(val, attr, None))
        except Exception:
            found = None
        if found:
            return found
    if watch is not None and id(val) in watch.buffers:
        return watch.buffers[id(val)]
    return "hbm" if name in params else None     # an argument of the kernel is the caller's tensor


def _where_text(buf, name, params):
    if buf is None:
        return ""
    if "hbm" in buf:
        return " in HBM (an input)" if name in params else " in HBM"
    return f" in {buf}"


_SAFE_NODES = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add,
               ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.Subscript, ast.Slice,
               ast.Tuple, ast.Attribute, ast.Call)
_SAFE_CALLS = {"min", "max", "int", "len", "abs"}


def _evaluate(node, frame):
    """Evaluate one operand expression in the kernel's own frame, or return None.

    Only expressions that cannot do anything are evaluated: names, numbers, arithmetic, slices,
    attribute reads and min/max. `a[i*128:(i+1)*128, :]` is; a call that allocates is not.
    """
    for n in ast.walk(node):
        if not isinstance(n, _SAFE_NODES):
            return None
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name)
                                            and n.func.id in _SAFE_CALLS and not n.keywords):
            return None
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            return None
    try:
        expr = ast.fix_missing_locations(ast.Expression(body=node))
        return eval(compile(expr, "<operand>", "eval"), frame.f_globals, dict(frame.f_locals))
    except Exception:
        return None


def _statement_node(source, lineno):
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.stmt):
                end = node.end_lineno or node.lineno
                if node.lineno <= lineno <= end and (best is None
                                                    or end - node.lineno <= best[0]):
                    best = (end - node.lineno, node)
    except SyntaxError:
        return None
    return best[1] if best else None


def _own_expressions(stmt):
    """The parts of a statement that are its own: for a loop or an `if`, the header and not the
    body, because the body is other lines."""
    if not hasattr(stmt, "body"):
        return [stmt]
    return [v for k, v in ast.iter_fields(stmt)
            if k not in ("body", "orelse", "finalbody", "handlers") and isinstance(v, ast.AST)]


def _rules(func, ops, limits):
    """Sentences for the rules these operands actually break. Empty when none is broken, so a
    rule is never recited at a kernel that obeys it. Each names which dimension must match
    which; none gives the shape to write."""
    pmax, smax, mmax = limits
    out = []
    shape = {k: v[1] for k, v in ops.items()}
    buf = {k: v[2] for k, v in ops.items()}
    if func in ("dma_copy", "tensor_copy") and shape.get("dst") and shape.get("src"):
        nd, ns = math.prod(shape["dst"]), math.prod(shape["src"])
        if nd != ns:
            out.append(f"The two sides of a {func} must hold the same number of elements, "
                       f"because it does not slice, reshape or broadcast: dst holds {nd} "
                       f"elements and src holds {ns}.")
    if func in ("dma_copy", "tensor_copy"):
        for k in ("dst", "src"):
            if buf.get(k) in ("sbuf", "psum") and shape.get(k) and shape[k][0] > pmax:
                out.append(f"A tile on the chip may have at most {pmax} rows, since its first "
                           f"dimension is the partition dimension: {k} has {shape[k][0]}.")
    if func == "nc_matmul" and shape.get("stationary") and shape.get("moving"):
        st, mv, dst = shape["stationary"], shape["moving"], shape.get("dst")
        if st[0] != mv[0]:
            out.append(f"The first dimension of stationary and of moving is the one nc_matmul "
                       f"contracts, so the two must be equal: stationary's is {st[0]} and "
                       f"moving's is {mv[0]}.")
        elif st[0] > pmax:
            out.append(f"One nc_matmul can contract at most {pmax}: the contraction (first) "
                       f"dimension here is {st[0]}.")
        if len(st) > 1 and st[1] > smax:
            out.append(f"The second dimension of stationary may be at most {smax}: it is "
                       f"{st[1]}.")
        if len(mv) > 1 and mv[1] > mmax:
            out.append(f"The second dimension of moving may be at most {mmax}: it is {mv[1]}.")
        if dst and len(st) == 2 and len(mv) == 2 and tuple(dst) != (st[1], mv[1]):
            out.append("The result of nc_matmul has stationary's second dimension as its first "
                       "and moving's second dimension as its second, and dst must be exactly "
                       f"that shape: dst is {dst}.")
    if func == "ndarray":
        dims = ops.get("shape", (None,) * 4)[3]
        try:
            dims = tuple(int(d) for d in dims)
        except Exception:
            dims = None
        # nl.ndarray(shape, dtype, buffer=nl.sbuf): leaving buffer out means a tile on the chip.
        where = _region_name(ops["buffer"][3]) if "buffer" in ops else "sbuf"
        if dims is not None and where in ("sbuf", "psum"):
            if len(dims) < 2:
                out.append(f"A tile on the chip needs at least 2 dimensions, the partition "
                           f"dimension first: this shape has {len(dims)}.")
            elif dims[0] > pmax:
                out.append(f"A tile on the chip may have at most {pmax} rows: this one would "
                           f"have {dims[0]}.")
    return out


def _allocation(source, name, before, inside):
    """(line, statement) of the last assignment to `name` above line `before`, in the function
    spanning `inside`. That is where a tile got its shape."""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name \
                    and inside[0] <= node.lineno < before and (best is None
                                                               or node.lineno > best):
                best = node.lineno
    except SyntaxError:
        return None
    return (best, statement_at(source, best, limit=140)) if best else None


def _earlier_use(source, name, before, inside):
    """(line, statement) of the first instruction above line `before` that used tile `name` as an
    operand. Execution got past it, so the shape the tile has is the shape that line needed."""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") in OPERANDS
                    and node.func.attr != "ndarray" and inside[0] <= node.lineno < before):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                root = arg
                while isinstance(root, ast.Subscript):
                    root = root.value
                if isinstance(root, ast.Name) and root.id == name \
                        and (best is None or node.lineno < best):
                    best = node.lineno
    except SyntaxError:
        return None
    return (best, statement_at(source, best, limit=140)) if best else None


def _origin(func, ops, limits, source, lineno, inside, params):
    """Sentences naming the tile that has to change and the line that gave it its shape.

    Measured on level 3, two runs out of two: told the failing line, the shapes on it and the
    rule, the model sent back the same tile. The line that FAILS is the nc_matmul; the line that
    is WRONG is ten lines above it, where the result tile was allocated, and "change exactly what
    the checker names" points at the first. So follow the tile back to where it was made, and say
    which of its dimensions is off, in which direction, and which operand it has to match. That
    is which thing is wrong and which way -- not the number to type.
    """
    pmax = limits[0]
    shape = {k: v[1] for k, v in ops.items()}
    text = {k: v[0] for k, v in ops.items()}
    blame = []                                   # (role, what is wrong with it)

    def size(s):
        return math.prod(s)

    if func == "nc_matmul" and shape.get("stationary") and shape.get("moving") and shape.get("dst"):
        st, mv, dst = shape["stationary"], shape["moving"], shape["dst"]
        if len(st) == 2 and len(mv) == 2 and tuple(dst) != (st[1], mv[1]):
            if len(dst) != 2:
                blame.append(("dst", f"it has {len(dst)} dimension(s) and needs 2"))
            else:
                parts = []
                for i, (have, want, src, word) in enumerate(
                        ((dst[0], st[1], text["stationary"], "first"),
                         (dst[1], mv[1], text["moving"], "second"))):
                    if have != want:
                        parts.append(f"its {word} dimension is {have}, too "
                                     f"{'small' if have < want else 'large'}: it must equal the "
                                     f"second dimension of `{src}`")
                blame.append(("dst", "; ".join(parts)))
    if func in ("dma_copy", "tensor_copy") and shape.get("dst") and shape.get("src"):
        if size(shape["dst"]) != size(shape["src"]):
            # The tile the kernel allocated is the one it can reshape; an input is what it is.
            for role, other in (("dst", "src"), ("src", "dst")):
                root = text[role].split("[", 1)[0]
                if root not in params or role == "dst":
                    way = "small" if size(shape[role]) < size(shape[other]) else "large"
                    blame.append((role, f"it holds {size(shape[role])} elements, too {way} for "
                                        f"the {size(shape[other])} on the other side"))
                    break
        for role in ("dst", "src"):
            if shape[role][0] > pmax and ops[role][2] in ("sbuf", "psum"):
                blame.append((role, f"it has {shape[role][0]} rows, too many: at most {pmax}"))
    out, done = [], set()
    for role, what in blame:
        root = text[role].split("[", 1)[0].strip()
        if not root.isidentifier() or root in done:
            continue
        done.add(root)
        made = _allocation(source, root, lineno, inside) if root not in params else None
        used = _earlier_use(source, root, lineno, inside) if func != "nc_matmul" else None
        if used:
            # Measured on level 3: told to resize a tile that an earlier copy had filled, the
            # model resized it, broke that copy, was told to resize it back, and went round.
            # A tile an earlier instruction already used has the shape THAT instruction needed.
            # It is being used for two things, and the second needs a tile of its own.
            out.append(f"`{root}` is the wrong size here: {what}. But do not resize it: line "
                       f"{used[0]}, `{used[1]}`, already ran with the shape it has, so it is "
                       f"being used for two different things. Leave `{root}` as it is and "
                       f"allocate a separate tile for this line, sized to match the other side.")
        elif made and "too many: at most" in what:
            # No number on the allocation line fixes a tile that is too tall. Saying "change
            # that line" would send the model to try one.
            out.append(f"`{root}` is too tall: {what}. It got its shape on line {made[0]}, "
                       f"`{made[1]}`. No single tile can have that many rows, so a different "
                       f"number there will not do: the data has to be handled in pieces of at "
                       f"most that many rows, in a loop.")
        elif made:
            out.append(f"The tile to change is `{root}`: {what}. It got its shape on line "
                       f"{made[0]}, `{made[1]}`, so that is the line to change, not this one.")
        else:
            out.append(f"The tile to change is `{root}`: {what}.")
    return out


def state(exc, path, source, watch=None, limit=700):
    """' At that line: ...' wrapped in MARK, or '' when there is nothing certain to say."""
    return state_and_rules(exc, path, source, watch, limit)[0]


def state_and_rules(exc, path, source, watch=None, limit=700, origin=False):
    """(text, number of rules stated). Shapes alone say what was there; a rule says what was
    wrong with it. Only the second can stand in for advice."""
    try:
        return _state(exc, path, source, watch, limit + (350 if origin else 0), origin)
    except Exception as e:
        ERRORS.append(f"state: {type(e).__name__}: {e}")
        return "", 0


def _state(exc, path, source, watch, limit, origin=False):
    hit = _hit(exc, path, source)
    stmt = _statement_node(source, hit[0]) if hit else None
    if stmt is None:
        return "", 0
    frame = hit[1]
    local = dict(frame.f_locals)
    params = set()
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == frame.f_code.co_name:
                params = {a.arg for a in node.args.args}
    except SyntaxError:
        pass

    # Every variable the statement mentions, in the order it mentions them.
    facts, seen = [], set()
    names = sorted((n for part in _own_expressions(stmt) for n in ast.walk(part)
                    if isinstance(n, ast.Name)), key=lambda n: (n.lineno, n.col_offset))
    for n in names:
        if n.id in seen or n.id not in local:
            continue
        seen.add(n.id)
        val = local[n.id]
        if isinstance(val, (types.ModuleType, types.FunctionType, type)) or isinstance(val, bool):
            continue
        shape = _shape(val)
        if shape is not None:
            facts.append(f"`{n.id}` is {shape}"
                         + _where_text(_buffer(val, n.id, watch, params), n.id, params))
        elif isinstance(val, (int, float)):
            facts.append(f"`{n.id}` = {val}")
        elif isinstance(val, tuple) and val and all(isinstance(v, int) for v in val):
            facts.append(f"`{n.id}` = {val}")

    # The operands of the instruction on that line, including ones that are slices.
    rules = []
    for call in sorted((c for part in _own_expressions(stmt) for c in ast.walk(part)
                        if isinstance(c, ast.Call) and getattr(c.func, "attr", "") in OPERANDS),
                       key=lambda c: (c.lineno, c.col_offset)):
        func = call.func.attr
        given = dict(zip(OPERANDS[func], call.args))
        given.update({k.arg: k.value for k in call.keywords if k.arg})
        ops = {}
        for role, node in given.items():
            if role not in OPERANDS[func]:
                continue
            text = " ".join((ast.get_source_segment(source, node) or "").split())
            val = _evaluate(node, frame)
            shape = _shape(val) if val is not None else None
            root = node
            while isinstance(root, ast.Subscript):
                root = root.value
            rname = root.id if isinstance(root, ast.Name) else ""
            # A slice lives wherever the tensor it was cut from lives.
            base = local.get(rname) if isinstance(node, ast.Subscript) else val
            buf = _buffer(base, rname, watch, params) if base is not None else None
            ops[role] = (text, shape, buf, val)
            if shape is not None and not isinstance(node, ast.Name) and func != "ndarray":
                facts.append(f"{role} `{text[:80]}` is {shape}" + _where_text(buf, rname, params))
        try:
            import nkibench
            limits = (nkibench.PMAX, nkibench.GEMM_STATIONARY_FMAX, nkibench.GEMM_MOVING_FMAX)
        except Exception:
            limits = (128, 128, 512)
        rules = _rules(func, ops, limits)
        if origin:
            inside = _function_spans(source).get(frame.f_code.co_name, (1, hit[0]))
            rules += _origin(func, ops, limits, source, hit[0], inside, params)
        break
    if not facts and not rules:
        return "", 0
    text = (" At that line: " + "; ".join(facts) + "." if facts else "") \
        + "".join(" " + r for r in rules)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return f"{MARK}{text}{MARK}", len(rules)


# ---------------------------------------------------------------- selftest
#
# A located line is only worth sending if it is the RIGHT line. So plant a bug whose failing
# statement is known in a kernel that is otherwise correct, and check that statement is the one
# reported. `expect` is a fragment of the statement that must be named, and `facts` are things
# that were true on that line and must be reported as they were.

PLANTS = [
    dict(name="tile too small for the copy", level=3, case=0,
         old="rhs_tile = nl.ndarray(rhs.shape,",
         new="rhs_tile = nl.ndarray((rhs.shape[0], rhs.shape[1] // 2),",
         expect="nisa.dma_copy(dst=rhs_tile, src=rhs)",
         facts=["`rhs_tile` is (128, 256)", "`rhs` is (128, 512)",
                "dst holds 32768 elements and src holds 65536"],
         origin=["The tile to change is `rhs_tile`", "too small",
                 "`rhs_tile = nl.ndarray((rhs.shape[0], rhs.shape[1] // 2),"]),
    dict(name="1-D tile", level=3, case=0,
         old="lhs_tile = nl.ndarray(lhsT.shape,",
         new="lhs_tile = nl.ndarray((lhsT.shape[0] * lhsT.shape[1],),",
         expect="lhs_tile = nl.ndarray(",
         facts=["`lhsT` is (128, 64)", "needs at least 2"]),
    dict(name="whole tensor in one tile", level=4, case=1,
         old="  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n",
         new="  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n"
             "  whole = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)\n"
             "  nisa.dma_copy(dst=whole, src=lhsT)\n",
         expect="nisa.dma_copy(dst=whole, src=lhsT)",
         facts=["`whole` is (256, 256)", "`lhsT` is (256, 256)"],
         origin=["`whole` is too tall", "too many", "`whole = nl.ndarray(lhsT.shape,",
                 "in pieces"]),
    dict(name="tile too small, call split over lines, inside three loops", level=4, case=1,
         old="rhs_tile = nl.ndarray((TILE_K, TILE_N),",
         new="rhs_tile = nl.ndarray((TILE_K, TILE_N // 2),",
         expect="nisa.dma_copy(dst=rhs_tile,",
         facts=["`rhs_tile` is (128, 256)", "`k` = 0", "`TILE_K` = 128",
                "dst holds 32768 elements and src holds 65536"],
         origin=["The tile to change is `rhs_tile`", "too small",
                 "`rhs_tile = nl.ndarray((TILE_K, TILE_N // 2),"]),
    dict(name="one tile used for two things", level=3, case=0,
         old="nisa.tensor_copy(dst=result_sbuf, src=result_psum)",
         new="nisa.tensor_copy(dst=rhs_tile, src=result_psum)",
         expect="nisa.tensor_copy(dst=rhs_tile, src=result_psum)",
         facts=["`rhs_tile` is (128, 512)", "`result_psum` is (64, 512)"],
         origin=["do not resize it", "`nisa.dma_copy(dst=rhs_tile, src=rhs)`",
                 "allocate a separate tile"]),
    # Not pass/fail. This asks the simulator a question: what does a result tile of the wrong
    # shape look like from outside? If the answer is a reshape error, then a kernel with no
    # reshape in it can produce one, and the line is the only thing that says where.
    dict(name="result tile of the wrong shape", level=3, case=0, question=True,
         old="result_psum = nl.ndarray(result.shape,",
         new="result_psum = nl.ndarray((1, M),",
         expect="nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)",
         facts=["`result_psum` is (1, 64)", "`lhs_tile` is (128, 64)", "`rhs_tile` is (128, 512)",
                "dst must be exactly that shape"],
         origin=["The tile to change is `result_psum`", "its first dimension is 1, too small",
                 "second dimension of `rhs_tile`", "`result_psum = nl.ndarray((1, M),"]),
]


def _run(source, level, case, tag, watch=True):
    """Simulate `source` on one shape of `level`. Returns (path, exception or None, watch, out)."""
    import nkibench
    spec = nkibench.LEVELS[level]
    path = f"/tmp/_diagnose_{tag}.py"
    with open(path, "w") as f:
        f.write(source)
    args, _ = nkibench.make_inputs(spec["shapes"][case], level)
    w = watching() if watch else None
    try:
        if w is None:
            out, _ = nkibench.simulate_and_count(nkibench.load_kernel(path, spec["entry"]), args)
        else:
            with w:
                out, _ = nkibench.simulate_and_count(nkibench.load_kernel(path, spec["entry"]),
                                                     args)
    except nkibench.NkiMissing:
        raise
    except Exception as e:
        return path, e, w, None
    return path, None, w, out


def selftest():
    import nkibench
    here = os.path.dirname(os.path.abspath(__file__))
    print("Planting known bugs in the reference kernels and checking the located line.\n")

    # statement_at needs no simulator, so prove it everywhere.
    src = "a = 1\nfor i in range(3):\n    f(a,\n      i)\n"
    ok = (statement_at(src, 3) == "f(a, i)" and statement_at(src, 4) == "f(a, i)"
          and statement_at(src, 2) == "for i in range(3):" and statement_at(src, 1) == "a = 1")
    print(f"  statement_at: split call whole, loop as its header -> {'ok' if ok else 'FAIL'}")
    marked = f"On K=1: {MARK}line 3, `x`, {MARK}raised E: m"
    ok2 = shown(marked) == "On K=1: line 3, `x`, raised E: m" and key(marked) == "On K=1: raised E: m"
    print(f"  shown / key: located text shown, then removed     -> {'ok' if ok2 else 'FAIL'}")
    rc = 0 if ok and ok2 else 1
    state_rc = origin_rc = 0

    try:
        import nki
    except ImportError:
        print("\n  NEEDS THE NEURON SDK: the plants below cannot run here. Run this in the seat pod.")
        return rc
    print(f"\n  simulator: nki {getattr(nki, '__version__', '?')} from {os.path.dirname(nki.__file__)}\n")

    # The control. If the untouched references do not run, a plant's failure proves nothing --
    # and watching a simulation must not change what it computes, or the watcher is the bug.
    import numpy as np
    refs = {}
    for level in sorted({p["level"] for p in PLANTS}):
        refs[level] = open(os.path.join(here, f"reference_level{level}.py")).read()
        for case in sorted({p["case"] for p in PLANTS if p["level"] == level}):
            _, err, _, plain = _run(refs[level], level, case, f"ref{level}", watch=False)
            _, err2, w, watched = _run(refs[level], level, case, f"ref{level}w", watch=True)
            lbl = nkibench.label(nkibench.LEVELS[level]["shapes"][case], level)
            print(f"  control: reference level {level} on {lbl} -> "
                  f"{'runs' if err is None else f'FAIL, raised {type(err).__name__}: {err}'}")
            same = err is None and err2 is None and np.array_equal(np.asarray(plain),
                                                                   np.asarray(watched))
            print(f"           and gives the same output while watched -> "
                  f"{'ok' if same else 'FAIL'}  ({len(w.buffers)} tile allocations seen)")
            rc |= 0 if err is None and same else 1
    print()

    for i, p in enumerate(PLANTS):
        ref = refs[p["level"]]
        if ref.count(p["old"]) != 1:
            print(f"  {p['name']}: FAIL, the reference no longer contains exactly one "
                  f"`{p['old'].strip()}` -- the plant is stale")
            rc |= 1
            continue
        source = ref.replace(p["old"], p["new"])
        path, err, w, _ = _run(source, p["level"], p["case"], f"plant{i}")
        if err is None:
            verdict = "the simulator accepted it" if p.get("question") else "FAIL, nothing was raised"
            print(f"  {p['name']}: {verdict}")
            rc |= 0 if p.get("question") else 1
            continue
        n = locate(err, path, source)
        stmt = statement_at(source, n) if n else ""
        right = bool(n) and p["expect"] in stmt
        if p.get("question"):
            print(f"  QUESTION, {p['name']}:")
        else:
            print(f"  {p['name']}: {'ok' if right else 'FAIL'}")
            rc |= 0 if right else 1
        print(f"      raised   {type(err).__name__}: {str(err)[:150]}")
        print(f"      located  {f'line {n}, `{stmt}`' if n else 'NO LINE FOUND'}")
        if not right and not p.get("question"):
            print(f"      expected a statement containing `{p['expect']}`")
        said = shown(state(err, path, source, w))
        missing = [f for f in p.get("facts", []) if f not in said]
        print(f"      state   {said.strip() or 'NOTHING REPORTED'}")
        if p.get("question"):
            print(f"      (facts not reported: {missing or 'none'})")
        else:
            print(f"      state reports every planted fact -> {'ok' if not missing else 'FAIL'}"
                  + (f"  missing: {missing}" if missing else ""))
            state_rc |= 0 if not missing else 1
        traced = shown(state_and_rules(err, path, source, w, origin=True)[0])
        lost = [f for f in p.get("origin", []) if f not in traced]
        if p.get("origin"):
            tail = traced.split("elements.", 1)[-1].strip() if "elements." in traced else traced
            print(f"      origin  {tail[-330:] if tail else 'NOTHING TRACED'}")
            print(f"      origin names the planted tile and line -> {'ok' if not lost else 'FAIL'}"
                  + (f"  missing: {lost}" if lost else ""))
            origin_rc |= 0 if not lost else 1

    print()
    print("Located lines: " + ("all are the planted ones." if rc == 0 else
                               "SOMETHING FAILED. Do not use --features locate until this passes."))
    print("State on the line: " + ("every planted fact was reported." if state_rc == 0 else
                                   "SOMETHING FAILED. Do not use --features state until this passes."))
    print("Origin of the tile: " + ("every planted line was traced." if origin_rc == 0 else
                                    "SOMETHING FAILED. Do not use --features origin until this passes."))
    return rc | (state_rc << 1) | (origin_rc << 2)


def replay(log_path, level=None, features=("locate", "state")):
    """Re-grade every distinct failing kernel in an attempt log with the features on.

    Two uses. Before a measured run: the features meet kernels a model really wrote, which are
    stranger than any plant, and this shows that they cope -- for nothing, since no model is
    called. After one: the "was told / would add" pairs are the evidence for what changed.
    """
    import json
    import agent
    agent.CANDIDATE_PATH = "/tmp/_replay_level{level}.py"   # never the path a live run is using
    groups = {}
    with open(log_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r.get("code") or "").strip() or r["reward"] >= 1.0 - 1e-9:
                continue
            if level is not None and r["level"] != level:
                continue
            groups.setdefault((r["level"], "".join(r["code"].split())), [r, 0])[1] += 1
    n = dict(kernels=0, crashed=0, located=0, state=0, differs=0)
    for (lv, _), (r, count) in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
        n["kernels"] += 1
        agent.FEATURES.clear()
        agent.FEATURES.update(features)
        try:
            _, _, marked = agent._grade(r["code"], lv)
        finally:
            agent.FEATURES.clear()
        added = [seg.strip() for seg in re.findall(f"{MARK}(.*?){MARK}", marked, re.S)]
        crashed = "raised " in key(marked)
        n["crashed"] += crashed
        n["located"] += any(seg.startswith("line ") for seg in added)
        n["state"] += any("At that line" in seg for seg in added)
        print(f"\nlevel {lv}   {count} attempt(s) were this kernel   reward {r['reward']:.2f}")
        print("  was told : " + " ".join((r.get("feedback") or "").split())[:300])
        if not r.get("features") and key(marked) != r.get("feedback"):
            n["differs"] += 1
            print("  NOTE     : grading it again gave a different message: "
                  + " ".join(key(marked).split())[:200])
        if added:
            for seg in added:
                print("  would add: " + seg)
        else:
            print("  would add: nothing" + ("" if crashed else
                                            " (it did not crash inside the kernel)"))
    print(f"\n{n['kernels']} distinct failing kernels; {n['crashed']} crashed in the simulator; "
          f"a line was named for {n['located']} of those and the state for {n['state']}.")
    if n["differs"]:
        print(f"{n['differs']} graded differently from the log. Grading should be repeatable: "
              f"look at those before trusting either.")
    print(f"errors inside the diagnosis itself: {len(ERRORS)}" + ("" if not ERRORS else
          "\n  " + "\n  ".join(sorted(set(ERRORS)))))
    return 1 if ERRORS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", metavar="LOG.jsonl")
    ap.add_argument("--level", type=int)
    ap.add_argument("--features", default="locate,state", help="for --replay")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.replay:
        sys.exit(replay(a.replay, a.level, {f for f in a.features.split(",") if f}))
    ap.print_help()


if __name__ == "__main__":
    main()
