"""Conservative symbol recognition, not a general mathematical expression parser."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_FUNCTIONS = frozenset({"ln", "log", "exp", "sin", "cos", "tan", "sqrt", "min", "max", "sum", "abs"})
_LETTERS = r"A-Za-z\u0370-\u03ff\u4e00-\u9fff"
_SUBSCRIPT = rf"[{_LETTERS}0-9]+(?:[_,][{_LETTERS}0-9]+)*"
_IDENTIFIER = re.compile(rf"[{_LETTERS}]+(?:_{_SUBSCRIPT})?[′’']*")
_SUM_START = re.compile(rf"(?<![{_LETTERS}0-9_])sum\s*\(")


def normalize_symbol(value: str) -> str:
    return re.sub(r"[\s{}]", "", value).replace("’", "′").replace("'", "′")


def _lexical_symbols(expression: str, declared: Iterable[str] = ()) -> set[str]:
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


def _sum_arguments(body: str, declared: set[str]) -> list[str] | None:
    """Split the explicit four-argument notation, protecting declared subscripts.

    Commas in P_i,j,k are part of the symbol. Unknown or ambiguous argument
    structure is left unbound rather than guessing an index from nearby text.
    """
    subscripts = sorted((s for s in declared if '_' in s), key=lambda s: (-len(s), s))
    parts: list[str] = []
    start = pos = 0
    stack: list[str] = []
    pairs = {')':'(', ']':'[', '}':'{'}
    while pos < len(body):
        char = body[pos]
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return None
        elif not stack:
            token = _IDENTIFIER.match(body, pos)
            if token:
                known = next((s for s in subscripts if body.startswith(s, pos)
                              and not re.match(rf'[{_LETTERS}0-9_′’\']', body[pos+len(s):pos+len(s)+1])), None)
                pos += len(known) if known else len(token.group())
                continue
            if char == ',':
                parts.append(body[start:pos].strip())
                start = pos + 1
        pos += 1
    parts.append(body[start:].strip())
    if stack or len(parts) != 4 or not all(parts):
        return None
    if not re.fullmatch(rf'[{_LETTERS}]+', parts[1]) or parts[1] in _FUNCTIONS:
        return None
    return parts


def formula_symbols(expression: str, declared: Iterable[str] = (), *,
                    exclude_bound_indices: bool = False) -> set[str]:
    """Recognize symbols; optionally omit indices bound by sum(term,index,lo,hi).

    Binding applies to the summand only, including nested sums. Bounds and
    occurrences outside that sum remain free. Indexed quantities keep their
    complete identity; an index i never removes the quantity x_i.
    """
    vocabulary = {normalize_symbol(s) for s in declared if s}
    expression = re.sub(r'_\{([^{}]*)\}', lambda m: '_' + normalize_symbol(m.group(1)), expression)

    def scan(text: str, bound: frozenset[str]) -> set[str]:
        found: set[str] = set()
        consumed = search = 0
        while match := _SUM_START.search(text, search):
            depth = 1
            end = match.end()
            while end < len(text) and depth:
                if text[end] == '(':
                    depth += 1
                elif text[end] == ')':
                    depth -= 1
                end += 1
            args = None if depth else _sum_arguments(text[match.end():end-1], vocabulary)
            if args is None:
                search = match.end()
                continue
            found.update(_lexical_symbols(text[consumed:match.start()], vocabulary) - bound)
            term, index, lower, upper = args
            term_bound = bound | {index} if exclude_bound_indices else bound
            found.update(scan(term, frozenset(term_bound)))
            found.update(scan(lower, bound))
            found.update(scan(upper, bound))
            if not exclude_bound_indices:
                found.add(index)
            consumed = search = end
        found.update(_lexical_symbols(text[consumed:], vocabulary) - bound)
        return found

    return scan(expression, frozenset())
