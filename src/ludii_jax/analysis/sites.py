"""
Site set evaluator for Ludii start position expressions.

Evaluates expressions like:
  difference expand sites Bottom steps: - 3 1 sites Phase 1
  union sites Bottom sites "A3"
  intersection sites Phase 0 union sites Top sites Bottom

Returns a set of cell indices.
"""

import re
from .topology import BoardTopology


def evaluate_sites(expr: str, topo: BoardTopology) -> set:
    """Evaluate a Ludii site set expression against a topology.

    Supports: sites Bottom/Top/Left/Right/Board/Empty/Outer/Phase N,
    expand(expr, steps:N), difference(A, B), union(A, B),
    intersection(A, B), specific cell indices, coord strings.
    """
    expr = expr.strip()
    n = topo.num_sites

    # Binary set ops: difference A B, union A B, intersection A B
    for op in ("difference", "union", "intersection"):
        if expr.startswith(op):
            rest = expr[len(op):].strip()
            a_expr, b_expr = _split_two_args(rest)
            a = evaluate_sites(a_expr, topo)
            b = evaluate_sites(b_expr, topo)
            if op == "difference":
                return a - b
            elif op == "union":
                return a | b
            else:
                return a & b

    # expand <inner> steps: <expr>
    if expr.startswith("expand"):
        rest = expr[len("expand"):].strip()
        # Find "steps:" to split inner from steps
        steps_match = re.search(r'steps:\s*', rest)
        if steps_match:
            inner = rest[:steps_match.start()].strip()
            steps_expr = rest[steps_match.end():].strip()
            steps = _eval_int_expr(steps_expr)
        else:
            inner = rest
            steps = 1
        base = evaluate_sites(inner, topo)
        return _expand(base, topo, steps)

    # sites Phase N — checkerboard phase based on (col + row) % 2
    m = re.match(r'sites\s+Phase\s+(\d+)', expr)
    if m:
        phase = int(m.group(1))
        return {i for i in range(n) if _site_phase(i, topo) == phase}

    # sites P1/P2 — player regions (use custom regions if defined, else Bottom/Top)
    m = re.match(r'sites\s+(P[12]|Mover|Next)', expr)
    if m:
        tag = m.group(1)
        # Check for custom region (from equipment regions definition)
        region_key = tag.lower()  # "p1" or "p2"
        if region_key in topo.regions:
            return {i for i in range(n) if topo.regions[region_key][i]}
        # Default: P1=Bottom, P2=Top
        region = "bottom" if tag in ("P1", "Mover") else "top"
        if region in topo.regions:
            return {i for i in range(n) if topo.regions[region][i]}
        return set()

    # sites Region
    m = re.match(r'sites\s+(Bottom|Top|Left|Right|Board|Empty|Outer|Centre|Center)', expr, re.IGNORECASE)
    if m:
        region = m.group(1).lower()
        if region in ("centre", "center"):
            region = "centre"
        if region == "board":
            return set(range(n))
        if region in topo.regions:
            return {i for i in range(n) if topo.regions[region][i]}
        # Fallback approximations
        quarter = max(n // 4, 1)
        if region == "bottom":
            return set(range(quarter))
        if region == "top":
            return set(range(n - quarter, n))
        if region == "left":
            cx = sum(x for x, _ in topo.site_coords) / n if n > 0 else 0
            return {i for i in range(n) if topo.site_coords[i][0] < cx}
        if region == "right":
            cx = sum(x for x, _ in topo.site_coords) / n if n > 0 else 0
            return {i for i in range(n) if topo.site_coords[i][0] > cx}
        if region == "centre":
            return {n // 2}
        return set(range(n))

    # sites Row N
    m = re.match(r'sites\s+Row\s+(\d+)', expr)
    if m:
        row = int(m.group(1))
        if topo.site_coords:
            return {i for i, (x, y) in enumerate(topo.site_coords) if int(round(y)) == row}
        return set()

    # sites Column N
    m = re.match(r'sites\s+Column\s+(\d+)', expr)
    if m:
        col = int(m.group(1))
        if topo.site_coords:
            return {i for i, (x, y) in enumerate(topo.site_coords) if int(round(x)) == col}
        return set()

    # Multi-coord: "A1" "C1" "E1" or single "A1"
    coords = re.findall(r'"([A-Za-z]\d+)"', expr)
    if coords:
        result = set()
        for coord in coords:
            col = ord(coord[0].upper()) - ord('A')
            row = int(coord[1:]) - 1
            if topo.site_coords:
                best = min(range(n), key=lambda i: abs(topo.site_coords[i][0] - col) + abs(topo.site_coords[i][1] - row))
                result.add(best)
        if result:
            return result

    # Explicit cell index
    if expr.isdigit():
        idx = int(expr)
        return {idx} if idx < n else set()

    # List of cell indices: 2 3 4 8 9 10
    tokens = expr.split()
    if tokens and all(t.isdigit() for t in tokens):
        return {int(t) for t in tokens if int(t) < n}

    # centrePoint
    if "centrePoint" in expr or "Centre" in expr:
        return {n // 2}

    # Fallback: try to find any recognizable sub-expression
    for kw in ["sites", "expand", "difference", "union", "intersection"]:
        idx = expr.find(kw)
        if idx > 0:
            return evaluate_sites(expr[idx:], topo)

    return set()


def _site_phase(cell_idx: int, topo: BoardTopology) -> int:
    """Compute the checkerboard phase of a cell: (col + row) % 2."""
    if topo.site_coords and cell_idx < len(topo.site_coords):
        x, y = topo.site_coords[cell_idx]
        return (int(round(x)) + int(round(y))) % 2
    return cell_idx % 2


def _eval_int_expr(expr: str) -> int:
    """Evaluate a simple integer expression like '- 3 1' → 2, or '2' → 2."""
    expr = expr.strip()
    # Prefix subtraction: - A B
    m = re.match(r'-\s+(\d+)\s+(\d+)', expr)
    if m:
        return int(m.group(1)) - int(m.group(2))
    # Prefix addition: + A B
    m = re.match(r'\+\s+(\d+)\s+(\d+)', expr)
    if m:
        return int(m.group(1)) + int(m.group(2))
    # Plain integer
    m = re.match(r'(\d+)', expr)
    if m:
        return int(m.group(1))
    return 1


def _expand(base: set, topo: BoardTopology, steps: int) -> set:
    """BFS expand a set of sites by N steps."""
    expanded = set(base)
    frontier = set(base)
    n = topo.num_sites
    for _ in range(steps):
        new_frontier = set()
        for idx in frontier:
            for d in range(topo.max_neighbors):
                nb = int(topo.adjacency[d, idx])
                if nb < n and nb not in expanded:
                    new_frontier.add(nb)
                    expanded.add(nb)
        frontier = new_frontier
    return expanded


def _split_two_args(text: str) -> tuple:
    """Split text into two arguments for binary set operations.

    Handles nested expressions by tracking depth through set-op keywords.
    """
    keywords = {"difference", "union", "intersection", "expand"}
    words = text.split()
    depth = 0

    for i, w in enumerate(words):
        if w in keywords:
            depth += 1
        # A "sites" keyword at depth 0 (after some content) starts arg 2
        if w == "sites" and i > 0 and depth <= 1:
            before = " ".join(words[:i])
            # Check if first arg looks complete
            if _looks_complete(before):
                return before, " ".join(words[i:])
        # Handle numeric-only second arg at depth boundary
        if depth == 0 and i > 0 and w.isdigit():
            before = " ".join(words[:i])
            if _looks_complete(before):
                return before, " ".join(words[i:])

    # Fallback: split at last "sites" keyword
    for i in range(len(words) - 1, 0, -1):
        if words[i] == "sites":
            return " ".join(words[:i]), " ".join(words[i:])

    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def _looks_complete(expr: str) -> bool:
    """Heuristic: does this look like a complete site set expression?"""
    region_kws = {"Bottom", "Top", "Left", "Right", "Board", "Phase", "Empty", "Outer", "Centre", "Row", "Column"}
    for kw in region_kws:
        if kw in expr:
            return True
    # Has a steps: expression
    if "steps:" in expr:
        return True
    # Has a coord
    if re.search(r'"[A-Za-z]\d+"', expr):
        return True
    # Has a digit at end (phase number, cell index, steps value)
    if re.search(r'\d\s*$', expr):
        return True
    return False
