"""Conservative symbol recognition, not a general mathematical expression parser."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_FUNCTIONS = frozenset({"ln", "log", "exp", "sin", "cos", "tan", "sqrt", "min", "max", "sum", "abs"})
_IDENTIFIER = re.compile(r"[A-Za-z\u0370-\u03ff\u4e00-\u9fff]+(?:_[A-Za-z0-9\u0370-\u03ff\u4e00-\u9fff_,]+)?[′’']*")


def normalize_symbol(value: str) -> str:
    return re.sub(r"[\s{}]", "", value).replace("’", "′").replace("'", "′")


def formula_symbols(expression: str, declared: Iterable[str] = ()) -> set[str]:
    """Keep case/subscripts; split an implicit product only if wholly declared.

    Longest declared identifiers win over shorter factors (TS is not T*S).
    Known functions are never split into variable letters. Unknown words are
    kept whole, so incomplete tokenization cannot turn `unknown` into `n`.
    """
    vocabulary = {normalize_symbol(s) for s in declared if s}
    vocabulary -= _FUNCTIONS
    pieces = sorted(vocabulary | _FUNCTIONS, key=lambda s: (-len(s), s))

    @lru_cache(maxsize=None)
    def split_product(token: str) -> tuple[str, ...] | None:
        if not token:
            return ()
        if token in vocabulary or token in _FUNCTIONS:
            return (token,)
        for piece in pieces:
            if token.startswith(piece):
                rest = split_product(token[len(piece):])
                if rest is not None:
                    return (piece, *rest)
        return None

    found: set[str] = set()
    # Compact only explicit braced subscripts; whitespace between identifiers
    # remains a boundary (e.g. a numerator followed by the next line's LHS).
    expression = re.sub(r"_\{([^{}]*)\}", lambda m: "_" + normalize_symbol(m.group(1)), expression)
    for match in _IDENTIFIER.finditer(expression):
        token = normalize_symbol(match.group())
        # Subscripts are integral to identity; never decompose x_ij as x_i*j.
        parts = (token,) if "_" in token else split_product(token) or (token,)
        found.update(part for part in parts if part not in _FUNCTIONS)
    return found
