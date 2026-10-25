"""Pick an instance per call shape: one launch per run of rows, gated on the rows' var."""


def gated(rows, runs):
    """`(value, when)` per launch, for `runs` of `(value, first, last)` rows covering a call's rows.

    Rows that are a var, or a var times a constant (the draft's `seqs*5`), gate
    on that var; a fixed row count needs one launch and no gate.
    """
    if isinstance(rows, int):
        return [next((v, None) for v, first, last in runs if first <= rows <= last)]
    var, scale = (rows, 1) if isinstance(rows, str) else rows["mul"]
    live = [(v, -(-first // scale), last // scale) for v, first, last in runs]
    live = [(v, lo, hi) for v, lo, hi in live if lo <= hi]
    return [(v, None if len(live) == 1 else {"var": var, **({"min": lo} if lo > 1 else {}), "max": hi})
            for v, lo, hi in live]


def meet(a, b):
    """The runs of rows over which both `a` and `b` hold one value, valued by the pair."""
    return [((va, vb), max(fa, fb), min(la, lb)) for va, fa, la in a for vb, fb, lb in b if max(fa, fb) <= min(la, lb)]
