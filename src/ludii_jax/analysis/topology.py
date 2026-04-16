"""
Board topology construction from Ludii board specifications.

Every board — square, hex, triangle, concentric, graph, merge — reduces to
a single BoardTopology: a set of sites with an adjacency matrix.

This is the foundation. Lines, slides, hops, connections all derive from
walking the adjacency graph. No board-type enum. One code path.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoardTopology:
    """Universal board representation for any game."""
    num_sites: int
    adjacency: np.ndarray          # int16[max_neighbors, num_sites] → neighbor index or num_sites (sentinel)
    max_neighbors: int
    site_coords: list              # [(x, y), ...] for rendering
    regions: dict = field(default_factory=dict)  # name → bool[num_sites]
    site_valid: np.ndarray = None  # bool[num_sites] — mask for irregular shapes

    def __post_init__(self):
        if self.site_valid is None:
            self.site_valid = np.ones(self.num_sites, dtype=bool)

    def get_side_cells(self, side_name: str) -> set:
        """Get cells on a named board side (N, S, E, W, NE, NW, SE, SW, Top, Bottom).

        For hex boards, uses angle-based classification from board center.
        For grid boards, uses coordinate extremes.
        """
        n = self.num_sites
        if not self.site_coords or n == 0:
            return set()

        all_x = [x for x, _ in self.site_coords]
        all_y = [y for _, y in self.site_coords]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        cx = sum(all_x) / n
        cy = sum(all_y) / n
        eps = 0.01

        # For hex boards: angle-based side classification
        if self.max_neighbors == 6:
            nb_counts = [sum(1 for d in range(self.max_neighbors)
                             if int(self.adjacency[d, i]) < n) for i in range(n)]
            max_nbs = max(nb_counts)
            edge_cells = [i for i in range(n) if nb_counts[i] < max_nbs]

            angle_ranges = {
                "S": (-120, -60), "SE": (-60, 0), "NE": (0, 60),
                "N": (60, 120), "NW": (120, 180), "SW": (-180, -120),
            }
            if side_name in angle_ranges:
                lo, hi = angle_ranges[side_name]
                cells = set()
                for i in edge_cells:
                    x, y = self.site_coords[i]
                    angle = math.atan2(y - cy, x - cx) * 180 / math.pi
                    if lo <= angle < hi or (side_name == "SW" and angle >= 180 - eps):
                        cells.add(i)
                return cells

        # Grid boards: coordinate-based
        cells = set()
        for i in range(n):
            x, y = self.site_coords[i]
            if side_name in ("N", "Top") and abs(y - max_y) < eps:
                cells.add(i)
            elif side_name in ("S", "Bottom") and abs(y - min_y) < eps:
                cells.add(i)
            elif side_name == "E" and abs(x - max_x) < eps:
                cells.add(i)
            elif side_name == "W" and abs(x - min_x) < eps:
                cells.add(i)
        return cells

    # Direction constants for 8-dir grids
    DIAG_DIRS = [1, 3, 5, 7]      # NE, SE, SW, NW
    ORTHO_DIRS = [0, 2, 4, 6]     # N, E, S, W
    P1_FWD_DIRS = [0, 1, 7]       # N, NE, NW
    P1_FWD_DIAG = [1, 7]          # NE, NW
    P2_FWD_DIRS = [4, 3, 5]       # S, SE, SW
    P2_FWD_DIAG = [3, 5]          # SE, SW


def build_topology(board_text: str) -> BoardTopology:
    """Parse a Ludii board description and return a BoardTopology.

    Handles: square, rectangle, hex, hexagon, tri, diamond,
    concentric, graph, complete, spiral, star,
    merge, add, remove, shift, scale, rotate, union.
    """
    tokens = board_text.replace("(", " ").replace(")", " ").split()
    if not tokens:
        return _grid(5, 5)

    shape = tokens[0].lower()
    use_vertex = "use:Vertex" in board_text or "use:vertex" in board_text

    if shape == "square":
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        n = args[0] if args else 8
        if use_vertex:
            has_diag = "diagonals:" in board_text.lower()
            return _vertex_grid(n, n, diagonals=has_diag)
        return _grid(n, n)

    if shape == "rectangle":
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        if len(args) >= 2:
            rows, cols = args[0], args[1]
            if use_vertex:
                has_diag = "diagonals:" in board_text.lower()
                return _vertex_grid(cols, rows, diagonals=has_diag)
            return _grid(cols, rows)
        return _grid(8, 8)

    if shape in ("hex", "hexagon"):
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        is_diamond = "diamond" in board_text.lower()
        is_triangle = "triangle" in board_text.lower()
        is_rectangle = "rectangle" in board_text.lower()
        if is_diamond and args:
            return _hex_diamond(args[0])
        if is_triangle and args:
            # Triangular hex: N rows with widths 1, 2, ..., N
            n = args[0]
            return _hex_variable(list(range(1, n + 1)))
        if is_rectangle and len(args) >= 2:
            # Hex rectangle: alternating row widths (cols, cols-1, cols, cols-1, ...)
            cols, rows = args[0], args[1]
            widths = [cols if r % 2 == 0 else cols - 1 for r in range(rows)]
            return _hex_variable(widths)
        if len(args) >= 3:
            return _hex_variable([int(a) for a in args])
        if args:
            return _hex_regular(args[0])
        return _hex_regular(5)

    if shape == "tri":
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        if len(args) >= 3:
            return _hex_variable([int(a) for a in args])
        if args:
            return _tri(args[0])
        return _tri(5)

    if shape == "diamond":
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        n = args[0] if args else 11
        return _hex_diamond(n)

    if shape == "concentric":
        return _concentric(board_text)

    if shape == "graph":
        return _graph_explicit(board_text)

    if shape == "complete":
        return _complete(board_text)

    if shape == "spiral":
        args = [int(t) for t in tokens[1:] if t.isdigit()]
        n = args[0] if args else 20
        return _spiral(n)

    # Wrapper operations (rotate, scale): extract the inner board
    if shape in ("rotate", "scale"):
        # Find the inner board definition and recurse
        inner = board_text
        for prefix in ["rotate", "scale"]:
            if inner.lower().startswith(prefix):
                # Skip the operation keyword and any numeric args
                rest = inner[len(prefix):].strip()
                while rest and (rest[0].isdigit() or rest[0] in '.-'):
                    rest = rest[1:].strip()
                inner = rest
                break
        if inner != board_text:
            return build_topology(inner)

    # Remove: build inner board, then remove specific cells
    if shape == "remove":
        # Extract cells to remove: cells:{N N N ...} or cells: N N N (braces stripped)
        cells_match = re.search(r'cells:[\s{]*([\d\s]+)', board_text)
        # Find inner board spec: everything between "remove" and "cells:"
        rest = board_text[len("remove"):].strip()
        if cells_match:
            # Inner is text before "cells:"
            cells_pos = rest.find("cells:")
            inner = rest[:cells_pos].strip() if cells_pos >= 0 else rest
            remove_cells = set(int(x) for x in re.findall(r'\d+', cells_match.group(1)))
        else:
            remove_cells = set()
        inner_topo = build_topology(inner)
        if remove_cells:
            return _remove_cells(inner_topo, remove_cells)
        return inner_topo

    # Composite operations: merge, add, shift, etc.
    if shape in ("merge", "add", "shift",
                 "union", "keep", "trim", "skew", "dual", "splitcrossings",
                 "renumber", "subdivide", "makefaces", "hole", "intersect",
                 "less"):
        return _composite(board_text)

    # Fallback: find largest recognizable sub-board
    for sn in ["square", "rectangle", "hex", "hexagon"]:
        matches = re.findall(rf'\b{sn}\s+(\d+)(?:\s+(\d+))?', board_text, re.IGNORECASE)
        if matches:
            best = max(matches, key=lambda m: int(m[0]))
            n1 = int(best[0])
            if sn in ("hex", "hexagon"):
                return _hex_regular(n1 if n1 % 2 == 1 else n1 + 1)
            if sn == "rectangle" and best[1]:
                return _grid(int(best[1]), n1)
            return _grid(n1, n1)

    # Last resort
    return _grid(7, 7)


# ============================================================
# Board constructors
# ============================================================

def _vertex_grid(width: int, height: int, diagonals: bool = False) -> BoardTopology:
    """Vertex grid: intersections of a rectangular grid.

    For (square N use:Vertex): width=N+1, height=N+1, 4 or 8 directions.
    Ludii convention: alternating diagonals for some games.
    """
    n = width * height
    coords = [(c, r) for r in range(height) for c in range(width)]

    if diagonals:
        # 8 directions: N, NE, E, SE, S, SW, W, NW (same as cell grid)
        offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        max_nb = 8
    else:
        # 4 orthogonal directions: N, E, S, W
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        max_nb = 4

    adj = np.full((max_nb, n), n, dtype=np.int16)
    for idx in range(n):
        r, c = idx // width, idx % width
        for d, (dr, dc) in enumerate(offsets):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                adj[d, idx] = nr * width + nc

    regions = {
        "bottom": np.array([i // width == 0 for i in range(n)]),
        "top": np.array([i // width == height - 1 for i in range(n)]),
        "left": np.array([i % width == 0 for i in range(n)]),
        "right": np.array([i % width == width - 1 for i in range(n)]),
    }

    return BoardTopology(n, adj, max_nb, coords, regions)


def _grid(width: int, height: int) -> BoardTopology:
    """Rectangular grid with 8 directions (ortho + diagonal).

    Matches Ludii cell numbering: row 0 = bottom, cell 0 = bottom-left.
    idx = row * width + col, where row increases upward.
    """
    n = width * height
    # Coords for rendering: (col, row) where row 0 is at bottom
    coords = [(c, r) for r in range(height) for c in range(width)]

    # 8 directions: N(+row), NE, E(+col), SE, S(-row), SW, W(-col), NW
    # In Ludii's bottom-up convention: N = increasing row index
    offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    adj = np.full((8, n), n, dtype=np.int16)

    for idx in range(n):
        r, c = idx // width, idx % width
        for d, (dr, dc) in enumerate(offsets):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                adj[d, idx] = nr * width + nc

    # Regions match Ludii: row 0 = bottom, row height-1 = top
    regions = {
        "bottom": np.array([i // width == 0 for i in range(n)]),
        "top": np.array([i // width == height - 1 for i in range(n)]),
        "left": np.array([i % width == 0 for i in range(n)]),
        "right": np.array([i % width == width - 1 for i in range(n)]),
    }

    # Centre region: the inner cells closest to the center of the board
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    centre = np.zeros(n, dtype=bool)
    for idx in range(n):
        r, c = idx // width, idx % width
        if abs(c - cx) < 1.0 and abs(r - cy) < 1.0:
            centre[idx] = True
    regions["centre"] = centre

    return BoardTopology(n, adj, 8, coords, regions)


def _hex_regular(side_length: int) -> BoardTopology:
    """Regular hexagonal board. Ludii hex N = side length N, total = 3N²-3N+1."""
    n_cells = 3 * side_length * side_length - 3 * side_length + 1
    radius = side_length - 1
    coords = []
    idx_map = {}

    for r in range(-radius, radius + 1):
        for q in range(-radius, radius + 1):
            if abs(r + q) <= radius:
                idx_map[(q, r)] = len(coords)
                x = q + r * 0.5
                y = r * math.sqrt(3) / 2
                coords.append((x, y))

    n = len(coords)
    assert n == n_cells, f"Hex {side_length}: expected {n_cells}, got {n}"
    # 6 hex axial directions: E, NE, NW, W, SW, SE  (dq, dr)
    hex_offsets = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    adj = np.full((6, n), n, dtype=np.int16)

    for (q, r), idx in idx_map.items():
        for d, (dq, dr) in enumerate(hex_offsets):
            nq, nr = q + dq, r + dr
            if (nq, nr) in idx_map:
                adj[d, idx] = idx_map[(nq, nr)]

    return BoardTopology(n, adj, 6, coords)


def _hex_variable(row_widths: list) -> BoardTopology:
    """Hex board with variable row widths, e.g. [3, 4, 3, 4, 3]."""
    coords = []
    row_starts = []

    for r, w in enumerate(row_widths):
        row_starts.append(len(coords))
        offset = (max(row_widths) - w) / 2
        for c in range(w):
            x = c + offset + (0.5 if r % 2 else 0)
            y = r * math.sqrt(3) / 2
            coords.append((x, y))

    n = len(coords)
    adj = np.full((6, n), n, dtype=np.int16)

    # Build adjacency by proximity
    for i in range(n):
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            dx = coords[j][0] - coords[i][0]
            dy = coords[j][1] - coords[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.5:
                neighbors.append(j)
        for d, nb in enumerate(neighbors[:6]):
            adj[d, i] = nb

    return BoardTopology(n, adj, 6, coords)


def _hex_diamond(n: int) -> BoardTopology:
    """Diamond-shaped hex board (n x n rhombus)."""
    coords = []
    idx_map = {}
    for r in range(n):
        for c in range(n):
            idx_map[(r, c)] = len(coords)
            x = c + r * 0.5
            y = r * math.sqrt(3) / 2
            coords.append((x, y))

    num_sites = len(coords)
    hex_offsets = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    adj = np.full((6, num_sites), num_sites, dtype=np.int16)

    for (r, c), idx in idx_map.items():
        for d, (dc, dr) in enumerate(hex_offsets):
            nr, nc = r + dr, c + dc
            if (nr, nc) in idx_map:
                adj[d, idx] = idx_map[(nr, nc)]

    return BoardTopology(num_sites, adj, 6, coords)


def _tri(size: int) -> BoardTopology:
    """Triangular grid, approximated as hex."""
    return _hex_regular(size if size % 2 == 1 else size + 1)


def _concentric(board_text: str) -> BoardTopology:
    """Concentric ring boards: (concentric Square rings:3), (concentric {1 6 6 6}).

    For Square rings:N, uses grid-coordinate layout matching Ludii's vertex ordering.
    Outer ring first, 8 vertices per ring: TL, T, TR, R, BR, B, BL, L clockwise.
    """
    tokens = board_text.replace("(", " ").replace(")", " ").split()
    nums = [int(t) for t in tokens if t.isdigit()]

    rings_match = re.search(r'rings:(\d+)', board_text)
    is_square = "square" in board_text.lower()

    if rings_match and is_square:
        rings = int(rings_match.group(1))
        # Grid-coordinate layout matching Ludii's row-by-row vertex numbering
        # For N rings on a 2N×2N grid, center at (N, N)
        grid_size = 2 * rings
        mid = rings  # center coordinate

        # Generate all ring vertices with grid coords
        all_verts = []  # (x, y)
        for r in range(rings):
            # Ring r: outer=0, inner=rings-1
            off = rings - r  # distance from center
            # 8 vertices: TL, T, TR, R, BR, B, BL, L
            verts = [
                (mid - off, mid + off), (mid, mid + off), (mid + off, mid + off),
                (mid + off, mid), (mid + off, mid - off), (mid, mid - off),
                (mid - off, mid - off), (mid - off, mid),
            ]
            all_verts.extend(verts)

        # Sort by Ludii convention: top-to-bottom (decreasing y), left-to-right (increasing x)
        indexed = [(x, y, i) for i, (x, y) in enumerate(all_verts)]
        indexed.sort(key=lambda v: (-v[1], v[0]))
        # Build mapping: old_idx → new_idx
        old_to_new = {}
        coords = []
        for new_idx, (x, y, old_idx) in enumerate(indexed):
            old_to_new[old_idx] = new_idx
            coords.append((float(x), float(y)))

        # Build edges using original ring structure, then remap
        edges = []
        for r in range(rings):
            base = r * 8
            # Ring edges: TL-T, T-TR, TR-R, R-BR, BR-B, B-BL, BL-L, L-TL
            for i in range(8):
                a, b = old_to_new[base + i], old_to_new[base + (i + 1) % 8]
                edges.append((min(a, b), max(a, b)))
            # Inter-ring: midpoints (T=1, R=3, B=5, L=7) connect to adjacent inner ring
            if r < rings - 1:
                inner_base = (r + 1) * 8
                for mid in [1, 3, 5, 7]:
                    a, b = old_to_new[base + mid], old_to_new[inner_base + mid]
                    edges.append((min(a, b), max(a, b)))

        # Deduplicate
        edges = list(set(edges))
        return _from_edges(coords, edges)

    # Non-square or explicit ring sizes
    if rings_match:
        rings = int(rings_match.group(1))
        sides = 4
        for t in tokens:
            if t.lower() == "triangle": sides = 3
            elif t.lower() == "hexagon": sides = 6
        ring_sizes = [2 * sides] * rings
    elif nums and len(nums) >= 2:
        ring_sizes = nums
    else:
        ring_sizes = [8, 8, 8]

    coords = []
    edges = []
    vertex_idx = 0
    ring_start_indices = []
    for ring_idx, ring_size in enumerate(ring_sizes):
        ring_start = vertex_idx
        ring_start_indices.append(ring_start)
        if ring_idx == 0 and ring_size == 1:
            coords.append((0.0, 0.0))
            vertex_idx += 1
            continue
        radius = ring_idx + 1
        for i in range(ring_size):
            angle = 2 * math.pi * i / ring_size
            coords.append((radius * math.cos(angle), radius * math.sin(angle)))
            if i > 0:
                edges.append((vertex_idx, vertex_idx - 1))
            vertex_idx += 1
        if ring_size > 1:
            edges.append((ring_start, vertex_idx - 1))
        if ring_idx > 0:
            inner_start = ring_start_indices[ring_idx - 1]
            inner_size = ring_sizes[ring_idx - 1]
            for i in range(ring_size):
                inner_i = (i * inner_size // ring_size) % inner_size if inner_size > 0 else 0
                edges.append((ring_start + i, inner_start + inner_i))

    return _from_edges(coords, edges)


def _graph_explicit(board_text: str) -> BoardTopology:
    """Parse (graph vertices: {x0 y0 x1 y1 ...} edges: {{v0 v1} ...})."""
    verts_match = re.search(r'vertices:?\s*([\d\s.\-]+?)(?:edges|$)', board_text)
    coords = []
    if verts_match:
        nums = re.findall(r'[\d.\-]+', verts_match.group(1))
        for i in range(0, len(nums) - 1, 2):
            coords.append((float(nums[i]), float(nums[i + 1])))

    edges_match = re.search(r'edges:?\s*(.*)', board_text, re.DOTALL)
    edges = []
    if edges_match:
        for v1, v2 in re.findall(r'(\d+)\s+(\d+)', edges_match.group(1)):
            edges.append((int(v1), int(v2)))

    if not coords:
        nums = re.findall(r'[\d.\-]+', board_text)
        for i in range(0, len(nums) - 1, 2):
            try:
                coords.append((float(nums[i]), float(nums[i + 1])))
            except ValueError:
                break

    return _from_edges(coords, edges)


def _complete(board_text: str) -> BoardTopology:
    """Complete graph or star polygon."""
    tokens = board_text.replace("(", " ").replace(")", " ").split()
    nums = [int(t) for t in tokens if t.isdigit()]
    n = nums[0] if nums else 5
    is_star = "star" in board_text.lower()

    coords = []
    edges = []

    if is_star:
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            coords.append((math.cos(angle), math.sin(angle)))
            inner_angle = angle + math.pi / n
            coords.append((0.5 * math.cos(inner_angle), 0.5 * math.sin(inner_angle)))
        for i in range(n):
            edges.append((2 * i, 2 * i + 1))
            edges.append((2 * i + 1, (2 * (i + 1)) % (2 * n)))
    else:
        for i in range(n):
            angle = 2 * math.pi * i / n
            coords.append((math.cos(angle), math.sin(angle)))
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

    return _from_edges(coords, edges)


def _spiral(n: int) -> BoardTopology:
    """Spiral track with n sites."""
    coords = []
    edges = []
    for i in range(n):
        angle = 4 * math.pi * i / n
        radius = 1 + i * 0.3
        coords.append((radius * math.cos(angle), radius * math.sin(angle)))
        if i > 0:
            edges.append((i - 1, i))
    return _from_edges(coords, edges)


def _composite(board_text: str) -> BoardTopology:
    """Parse composite boards (merge, add, remove, shift, etc.)."""
    coords = []
    edges = []

    # Extract all sub-board definitions with shifts
    for m in re.finditer(r'(?:shift\s+([\d.\-]+)\s+([\d.\-]+)\s+)?(?:square|rectangle)\s+(\d+)(?:\s+(\d+))?', board_text):
        dx = float(m.group(1)) if m.group(1) else 0
        dy = float(m.group(2)) if m.group(2) else 0
        n1 = int(m.group(3))
        n2 = int(m.group(4)) if m.group(4) else n1

        base_idx = len(coords)
        for row in range(n2):
            for col in range(n1):
                coords.append((dx + col, dy + row))
                idx = base_idx + row * n1 + col
                if col < n1 - 1:
                    edges.append((idx, idx + 1))
                if row < n2 - 1:
                    edges.append((idx, idx + n1))

    if coords:
        # Merge overlapping vertices
        return _merge_vertices(coords, edges)

    # Fallback: find largest number
    nums = [int(t) for t in board_text.split() if t.isdigit() and 3 <= int(t) <= 20]
    size = max(nums) if nums else 7
    return _grid(size, size)


# ============================================================
# Helpers
def _remove_cells(topo: BoardTopology, remove: set) -> BoardTopology:
    """Remove specific cells from a topology, renumbering remaining cells."""
    n = topo.num_sites
    keep = [i for i in range(n) if i not in remove]
    new_n = len(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}

    coords = [topo.site_coords[i] for i in keep]
    adj = np.full((topo.max_neighbors, new_n), new_n, dtype=np.int16)
    for new_idx, old_idx in enumerate(keep):
        for d in range(topo.max_neighbors):
            old_nb = int(topo.adjacency[d, old_idx])
            if old_nb < n and old_nb in old_to_new:
                adj[d, new_idx] = old_to_new[old_nb]

    regions = {}
    for name, mask in topo.regions.items():
        new_mask = np.array([mask[i] for i in keep])
        regions[name] = new_mask

    return BoardTopology(new_n, adj, topo.max_neighbors, coords, regions)


# ============================================================

def _from_edges(coords: list, edges: list) -> BoardTopology:
    """Build BoardTopology from vertex coordinates and edge list."""
    n = len(coords)
    if n == 0:
        return _grid(5, 5)

    # Build adjacency lists
    adj_lists = [[] for _ in range(n)]
    for v1, v2 in edges:
        if 0 <= v1 < n and 0 <= v2 < n:
            if v2 not in adj_lists[v1]:
                adj_lists[v1].append(v2)
            if v1 not in adj_lists[v2]:
                adj_lists[v2].append(v1)

    max_neighbors = max((len(a) for a in adj_lists), default=1)
    max_neighbors = max(max_neighbors, 1)

    adj = np.full((max_neighbors, n), n, dtype=np.int16)
    for v in range(n):
        for i, nb in enumerate(adj_lists[v][:max_neighbors]):
            adj[i, v] = nb

    return BoardTopology(n, adj, max_neighbors, coords)


def _merge_vertices(coords: list, edges: list, threshold: float = 0.1) -> BoardTopology:
    """Merge overlapping vertices from composite boards."""
    unique = []
    vert_map = {}

    for i, (x, y) in enumerate(coords):
        key = (round(x, 1), round(y, 1))
        if key not in vert_map:
            vert_map[key] = len(unique)
            unique.append((x, y))

    # Remap edges
    remapped = set()
    for v1, v2 in edges:
        k1 = (round(coords[v1][0], 1), round(coords[v1][1], 1))
        k2 = (round(coords[v2][0], 1), round(coords[v2][1], 1))
        nv1, nv2 = vert_map[k1], vert_map[k2]
        if nv1 != nv2:
            remapped.add((min(nv1, nv2), max(nv1, nv2)))

    # Add adjacency for nearby merged vertices
    for i, (x1, y1) in enumerate(unique):
        for j, (x2, y2) in enumerate(unique):
            if i >= j:
                continue
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < 1.5:
                remapped.add((i, j))

    return _from_edges(unique, list(remapped))
