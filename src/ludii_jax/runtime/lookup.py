"""
Precomputed lookup tables derived from BoardTopology.

All movement, line detection, and connectivity checks use these tables.
Computed once at compile time, baked into JIT as constants.
"""

import numpy as np
import jax.numpy as jnp

from ..analysis.topology import BoardTopology


def build_adjacency_lookup(topo: BoardTopology) -> jnp.ndarray:
    """(max_neighbors, num_sites, num_sites) boolean adjacency array."""
    n = topo.num_sites
    adj = np.zeros((topo.max_neighbors, n, n), dtype=np.int8)
    for d in range(topo.max_neighbors):
        for v in range(n):
            nb = int(topo.adjacency[d, v])
            if nb < n:
                adj[d, v, nb] = 1
    return jnp.array(adj)


def build_slide_lookup(topo: BoardTopology) -> jnp.ndarray:
    """(max_neighbors, num_sites, max_distance) → destination cell.

    slide_lookup[direction, start, distance] = the site reached by walking
    `distance` steps from `start` in `direction`. Sentinel = num_sites.
    """
    n = topo.num_sites
    max_dist = min(n, 32)  # cap at 32 for memory
    lookup = np.full((topo.max_neighbors, n, max_dist), n, dtype=np.int16)

    for d in range(topo.max_neighbors):
        for start in range(n):
            pos = start
            for dist in range(max_dist):
                lookup[d, start, dist] = pos
                next_pos = int(topo.adjacency[d, pos])
                if next_pos >= n:
                    break
                pos = next_pos

    return jnp.array(lookup)


def build_hop_between_lookup(topo: BoardTopology, slide_lookup: jnp.ndarray) -> jnp.ndarray:
    """(num_sites, num_sites) → the cell hopped over.

    hop_between[start, dest] = between cell, or num_sites (sentinel).
    """
    n = topo.num_sites
    lookup = np.full((n, n), n, dtype=np.int16)

    for d in range(topo.max_neighbors):
        for start in range(n):
            between = int(slide_lookup[d, start, 1])
            dest = int(slide_lookup[d, start, 2])
            if between < n and dest < n:
                lookup[start, dest] = between

    return jnp.array(lookup)


def build_line_indices(topo: BoardTopology, length: int) -> jnp.ndarray:
    """All lines of `length` consecutive sites following adjacency edges.

    Uses direction-walk for grid/hex boards. Falls back to coordinate-based
    collinear detection for edge-based boards (concentric, graph).
    Returns (num_lines, length) array of site indices.
    """
    n = topo.num_sites
    lines = set()

    # Direction-based: walk in fixed direction d for `length` steps
    for d in range(topo.max_neighbors):
        for start in range(n):
            path = [start]
            pos = start
            for _ in range(length - 1):
                next_pos = int(topo.adjacency[d, pos])
                if next_pos >= n or next_pos in path:
                    break
                path.append(next_pos)
                pos = next_pos
            if len(path) == length:
                lines.add(tuple(path))

    # Coordinate-based fallback: find collinear connected groups
    # This catches lines on concentric/graph boards where direction indices
    # don't maintain geometric consistency
    if topo.site_coords:
        from collections import defaultdict
        by_x = defaultdict(list)
        by_y = defaultdict(list)
        for i, (x, y) in enumerate(topo.site_coords):
            by_x[round(x, 2)].append(i)
            by_y[round(y, 2)].append(i)

        nbs_cache = {}
        def get_nbs(cell):
            if cell not in nbs_cache:
                nbs_cache[cell] = {int(topo.adjacency[d, cell]) for d in range(topo.max_neighbors) if int(topo.adjacency[d, cell]) < n}
            return nbs_cache[cell]

        for group in list(by_x.values()) + list(by_y.values()):
            if len(group) < length:
                continue
            # Sort by the varying coordinate
            if len(set(round(topo.site_coords[i][0], 2) for i in group)) == 1:
                group = sorted(group, key=lambda i: topo.site_coords[i][1])
            else:
                group = sorted(group, key=lambda i: topo.site_coords[i][0])
            # Find connected consecutive subsequences of `length`
            for start in range(len(group) - length + 1):
                sub = group[start:start + length]
                connected = all(sub[j+1] in get_nbs(sub[j]) for j in range(length - 1))
                if connected:
                    lines.add(tuple(sub))

    # Filter: only keep geometrically collinear lines (all cells on a straight line)
    if topo.site_coords and lines:
        filtered = set()
        for line in lines:
            coords_l = [topo.site_coords[c] for c in line]
            xs = [round(x, 2) for x, _ in coords_l]
            ys = [round(y, 2) for _, y in coords_l]
            # Collinear: all same x, all same y, or consistent slope
            if len(set(xs)) == 1 or len(set(ys)) == 1:
                filtered.add(line)
            elif length == 2:
                filtered.add(line)  # 2-cell lines are always "straight"
            else:
                # Check slope consistency for diagonal lines
                dx = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
                dy = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
                if all(abs(dx[i] - dx[0]) < 0.01 for i in range(len(dx))) and \
                   all(abs(dy[i] - dy[0]) < 0.01 for i in range(len(dy))):
                    filtered.add(line)
        lines = filtered

    if not lines:
        return jnp.zeros((0, length), dtype=jnp.int16)
    return jnp.array(sorted(lines), dtype=jnp.int16)


def build_edge_mask(topo: BoardTopology) -> jnp.ndarray:
    """Boolean mask: True for sites on the board boundary (fewer than max neighbors)."""
    n = topo.num_sites
    neighbor_counts = (topo.adjacency < n).sum(axis=0)
    max_count = neighbor_counts.max()
    return jnp.array(neighbor_counts < max_count)


def build_region_masks(topo: BoardTopology) -> dict:
    """Convert topology regions to JAX boolean masks."""
    masks = {}
    for name, arr in topo.regions.items():
        masks[name] = jnp.array(arr, dtype=jnp.bool_)
    return masks
