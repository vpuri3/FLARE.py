"""Laplacian eigenfeature operators (LOBPCG / FEM)."""

from __future__ import annotations

import numpy as np
import torch

from pdebench.dataset.laplacian.spec import parse_laplacian_spec


def _select_nonconstant_eigenvectors(eigenvectors: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    batch_size, num_nodes, _ = eigenvectors.shape
    k = int(num_eigenvectors)
    start = 1 if num_nodes > 1 else 0
    out = eigenvectors[:, :, start: start + min(k, max(num_nodes - start, 0))]
    if out.shape[-1] < k:
        out = torch.cat([out, out.new_zeros(batch_size, num_nodes, k - out.shape[-1])], dim=-1)
    return out


def _select_nonconstant_eigenpairs(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    num_eigenvectors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_nodes, _ = eigenvectors.shape
    k = int(num_eigenvectors)
    start = 1 if num_nodes > 1 else 0
    width = min(k, max(num_nodes - start, 0), max(int(eigenvalues.shape[-1]) - start, 0))
    values = eigenvalues[:, start: start + width]
    vectors = eigenvectors[:, :, start: start + width]
    if width < k:
        values = torch.cat([values, values.new_zeros(batch_size, k - width)], dim=-1)
        vectors = torch.cat([vectors, vectors.new_zeros(batch_size, num_nodes, k - width)], dim=-1)
    return values, vectors


def _unique_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        empty = torch.empty((0,), device=edge_index.device, dtype=torch.long)
        return empty, empty
    src, dst = edge_index.long()
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes) & (src != dst)
    src = src[valid]
    dst = dst[valid]
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pairs = torch.unique(torch.stack([lo, hi], dim=0), dim=1)
    return pairs[0], pairs[1]


def _sparse_normalized_laplacian(num_nodes: int, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    src, dst = _unique_undirected_edges(edge_index.to(device=device), num_nodes)
    if num_nodes == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.float32),
            (0, 0),
            device=device,
        ).coalesce()
    if src.numel() == 0:
        raise ValueError("Sparse normalized Laplacian requires at least one valid non-self edge.")

    vals = torch.ones(src.shape[0], device=device, dtype=torch.float32)
    rows = torch.cat([src, dst], dim=0)
    cols = torch.cat([dst, src], dim=0)
    vals = torch.cat([vals, vals], dim=0)
    adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (num_nodes, num_nodes), device=device).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    inv_sqrt_degree = torch.where(degree > 0, torch.rsqrt(degree), torch.zeros_like(degree))
    idx = adj.indices()
    off_vals = -inv_sqrt_degree[idx[0]] * adj.values() * inv_sqrt_degree[idx[1]]
    diag = torch.arange(num_nodes, device=device)
    lap_idx = torch.cat([torch.stack([diag, diag]), idx], dim=1)
    lap_vals = torch.cat([torch.ones(num_nodes, device=device, dtype=torch.float32), off_vals], dim=0)
    return torch.sparse_coo_tensor(lap_idx, lap_vals, (num_nodes, num_nodes), device=device).coalesce()


def _lobpcg_residual(operator: torch.Tensor, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> float:
    if not operator.is_sparse:
        raise ValueError("LOBPCG residual expects a sparse operator.")
    ax = torch.sparse.mm(operator, eigvecs)
    resid = ax - eigvecs * eigvals.unsqueeze(0)
    return float(torch.linalg.vector_norm(resid, dim=0).max().item())


def _prepare_lobpcg_init(
    init: torch.Tensor | None,
    current_vecs: torch.Tensor | None,
    num_nodes: int,
    stage_k: int,
    device: torch.device,
) -> torch.Tensor:
    if current_vecs is not None:
        base = current_vecs
    else:
        base = init
    if base is None:
        return torch.randn(num_nodes, stage_k, device=device, dtype=torch.float32)
    base = base.to(device=device, dtype=torch.float32)
    if base.shape[1] >= stage_k:
        return base[:, :stage_k]
    extra = torch.randn(num_nodes, stage_k - base.shape[1], device=device, dtype=torch.float32)
    return torch.cat([base, extra], dim=1)


def _lobpcg_ladder(
    operator: torch.Tensor,
    num_eigenvectors: int,
    *,
    ladder=(16, 33),
    niter=60,
    tol=1e-4,
    init: torch.Tensor | None = None,
    check_residual: bool = False,
    residual_tol: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not operator.is_sparse:
        raise ValueError("LOBPCG ladder expects a sparse operator.")
    num_nodes = int(operator.shape[0])
    target = min(num_nodes, int(num_eigenvectors) + 1 if num_nodes > 1 else 1)
    ladder = [min(int(step), target) for step in ladder if int(step) > 0]
    if not ladder or ladder[-1] != target:
        ladder.append(target)
    if num_nodes < 3 * target:
        dense = operator.to_dense()
        eigvals, eigvecs = torch.linalg.eigh(dense)
        return eigvals[:target], eigvecs[:, :target]

    current_vals = None
    current_vecs = None
    for stage_k in ladder:
        stage_init = _prepare_lobpcg_init(init, current_vecs, num_nodes, stage_k, operator.device)
        eigvals, eigvecs = torch.lobpcg(
            operator,
            k=stage_k,
            X=stage_init,
            largest=False,
            niter=int(niter),
            tol=tol,
            method="ortho",
        )
        if check_residual and _lobpcg_residual(operator, eigvals, eigvecs) > float(residual_tol):
            eigvals, eigvecs = torch.lobpcg(
                operator,
                k=stage_k,
                X=eigvecs.detach(),
                largest=False,
                niter=int(niter),
                tol=tol,
                method="ortho",
            )
        order = torch.argsort(eigvals)
        current_vals = eigvals[order]
        current_vecs = eigvecs[:, order]
    return current_vals, current_vecs


def _sparse_laplacian_eigenvectors_one(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    device = edge_index.device
    if num_nodes == 0:
        return torch.empty((0, int(num_eigenvectors)), device=device, dtype=torch.float32)
    if num_nodes == 1:
        return torch.ones((1, int(num_eigenvectors)), device=device, dtype=torch.float32)
    if edge_index.numel() == 0:
        raise ValueError("Graph Laplacian eigenvectors require non-empty mesh/graph connectivity.")
    operator = _sparse_normalized_laplacian(num_nodes, edge_index, device=device)
    _, eigvecs = _lobpcg_ladder(
        operator,
        num_eigenvectors,
        ladder=(16, int(num_eigenvectors) + 1),
        niter=60,
        tol=1e-4,
        init=init_eigenvectors,
    )
    return _select_nonconstant_eigenvectors(eigvecs.unsqueeze(0), num_eigenvectors)[0]


def _sparse_laplacian_eigenpairs_one(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    k = int(num_eigenvectors)
    if num_nodes == 0:
        return (
            torch.empty((k,), device=device, dtype=torch.float32),
            torch.empty((0, k), device=device, dtype=torch.float32),
        )
    if num_nodes == 1:
        return (
            torch.zeros((k,), device=device, dtype=torch.float32),
            torch.ones((1, k), device=device, dtype=torch.float32),
        )
    if edge_index.numel() == 0:
        raise ValueError("Graph Laplacian eigendecomposition requires non-empty mesh/graph connectivity.")
    operator = _sparse_normalized_laplacian(num_nodes, edge_index, device=device)
    eigvals, eigvecs = _lobpcg_ladder(
        operator,
        k,
        ladder=(16, k + 1),
        niter=60,
        tol=1e-4,
        init=init_eigenvectors,
    )
    vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigvecs.unsqueeze(0), k)
    return vals[0], vecs[0]


def _batched_laplacian_eigenvectors(
    num_nodes: int,
    edge_indices: list[torch.Tensor],
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = len(edge_indices)
    if batch_size == 0:
        raise ValueError("Cannot compute batched Laplacian eigenvectors for an empty graph list.")
    k = int(num_eigenvectors)
    device = edge_indices[0].device
    if num_nodes == 0:
        return torch.empty((batch_size, 0, k), device=device)
    eigenvectors = [
        _sparse_laplacian_eigenvectors_one(
            edge_index,
            num_nodes,
            k,
            init_eigenvectors=(
                init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None
            ),
        )
        for i, edge_index in enumerate(edge_indices)
    ]
    return torch.stack(eigenvectors, dim=0)


def _batched_laplacian_eigenpairs(
    num_nodes: int,
    edge_indices: list[torch.Tensor],
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(edge_indices)
    if batch_size == 0:
        raise ValueError("Cannot compute batched Laplacian eigenpairs for an empty graph list.")
    k = int(num_eigenvectors)
    device = edge_indices[0].device
    if num_nodes == 0:
        return torch.empty((batch_size, k), device=device), torch.empty((batch_size, 0, k), device=device)
    pairs = [
        _sparse_laplacian_eigenpairs_one(
            edge_index,
            num_nodes,
            k,
            init_eigenvectors=(
                init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None
            ),
        )
        for i, edge_index in enumerate(edge_indices)
    ]
    values, vectors = zip(*pairs, strict=True)
    return torch.stack(list(values), dim=0), torch.stack(list(vectors), dim=0)


def _batched_edge_weighted_laplacian_eigenvectors(
    num_nodes: int,
    edge_indices: list[torch.Tensor],
    positions: list[torch.Tensor],
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = len(edge_indices)
    if batch_size == 0:
        raise ValueError("Cannot compute edge-weighted Laplacian eigenvectors for an empty graph list.")
    k = int(num_eigenvectors)
    device = edge_indices[0].device
    if num_nodes == 0:
        return torch.empty((batch_size, 0, k), device=device)
    outputs = []
    for i, (edge_index, pos) in enumerate(zip(edge_indices, positions, strict=True)):
        if edge_index.numel() == 0:
            raise ValueError("Edge-weighted Laplacian eigenvectors require non-empty mesh/graph connectivity.")
        src, dst = _unique_undirected_edges(edge_index.to(device=device), num_nodes)
        if src.numel() == 0:
            raise ValueError("Edge-weighted Laplacian eigenvectors require at least one valid non-self edge.")
        pos = pos.to(device=device, dtype=torch.float32)
        length = torch.linalg.vector_norm(pos.index_select(0, src) - pos.index_select(0, dst), dim=-1).clamp_min(1e-8)
        weight = 1.0 / length
        rows = torch.cat([src, dst], dim=0)
        cols = torch.cat([dst, src], dim=0)
        vals = torch.cat([weight, weight], dim=0)
        adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (num_nodes, num_nodes), device=device).coalesce()
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        inv_sqrt_degree = torch.where(degree > 0, torch.rsqrt(degree), torch.zeros_like(degree))
        idx = adj.indices()
        off_vals = -inv_sqrt_degree[idx[0]] * adj.values() * inv_sqrt_degree[idx[1]]
        diag = torch.arange(num_nodes, device=device)
        lap_idx = torch.cat([torch.stack([diag, diag]), idx], dim=1)
        lap_vals = torch.cat([torch.ones(num_nodes, device=device, dtype=torch.float32), off_vals], dim=0)
        operator = torch.sparse_coo_tensor(lap_idx, lap_vals, (num_nodes, num_nodes), device=device).coalesce()
        _, eigenvectors = _lobpcg_ladder(
            operator,
            k,
            ladder=(16, k + 1),
            niter=60,
            tol=1e-4,
            init=init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None,
        )
        outputs.append(_select_nonconstant_eigenvectors(eigenvectors.unsqueeze(0), k)[0])
    return torch.stack(outputs, dim=0)


def _batched_edge_weighted_laplacian_eigenpairs(
    num_nodes: int,
    edge_indices: list[torch.Tensor],
    positions: list[torch.Tensor],
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(edge_indices)
    if batch_size == 0:
        raise ValueError("Cannot compute edge-weighted Laplacian eigenpairs for an empty graph list.")
    k = int(num_eigenvectors)
    device = edge_indices[0].device
    if num_nodes == 0:
        return torch.empty((batch_size, k), device=device), torch.empty((batch_size, 0, k), device=device)
    values_out = []
    vectors_out = []
    for i, (edge_index, pos) in enumerate(zip(edge_indices, positions, strict=True)):
        if edge_index.numel() == 0:
            raise ValueError("Edge-weighted Laplacian eigendecomposition requires non-empty mesh/graph connectivity.")
        src, dst = _unique_undirected_edges(edge_index.to(device=device), num_nodes)
        if src.numel() == 0:
            raise ValueError("Edge-weighted Laplacian eigendecomposition requires at least one valid non-self edge.")
        pos = pos.to(device=device, dtype=torch.float32)
        length = torch.linalg.vector_norm(pos.index_select(0, src) - pos.index_select(0, dst), dim=-1).clamp_min(1e-8)
        weight = 1.0 / length
        rows = torch.cat([src, dst], dim=0)
        cols = torch.cat([dst, src], dim=0)
        vals = torch.cat([weight, weight], dim=0)
        adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (num_nodes, num_nodes), device=device).coalesce()
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        inv_sqrt_degree = torch.where(degree > 0, torch.rsqrt(degree), torch.zeros_like(degree))
        idx = adj.indices()
        off_vals = -inv_sqrt_degree[idx[0]] * adj.values() * inv_sqrt_degree[idx[1]]
        diag = torch.arange(num_nodes, device=device)
        lap_idx = torch.cat([torch.stack([diag, diag]), idx], dim=1)
        lap_vals = torch.cat([torch.ones(num_nodes, device=device, dtype=torch.float32), off_vals], dim=0)
        operator = torch.sparse_coo_tensor(lap_idx, lap_vals, (num_nodes, num_nodes), device=device).coalesce()
        eigvals, eigvecs = _lobpcg_ladder(
            operator,
            k,
            ladder=(16, k + 1),
            niter=60,
            tol=1e-4,
            init=init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None,
        )
        vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigvecs.unsqueeze(0), k)
        values_out.append(vals[0])
        vectors_out.append(vecs[0])
    return torch.stack(values_out, dim=0), torch.stack(vectors_out, dim=0)


def _triangulate_cell(cell: torch.Tensor) -> list[tuple[int, int, int]]:
    unique = []
    for value in cell.tolist():
        ivalue = int(value)
        if ivalue not in unique:
            unique.append(ivalue)
    if len(unique) < 3:
        return []
    return [(unique[0], unique[i], unique[i + 1]) for i in range(1, len(unique) - 1)]


def _cells_to_triangles_numpy(cells: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Vectorized triangle extraction for triangle/quad cell arrays."""
    arr = cells.detach().to(device="cpu", dtype=torch.long).numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    nv = int(arr.shape[1])
    if nv == 3:
        tri = arr
    elif nv == 4:
        tri = np.concatenate([arr[:, [0, 1, 2]], arr[:, [0, 2, 3]]], axis=0)
    else:
        triangles: list[tuple[int, int, int]] = []
        for cell in arr:
            triangles.extend(_triangulate_cell(torch.from_numpy(cell)))
        if not triangles:
            return np.empty((0, 3), dtype=np.int64)
        tri = np.asarray(triangles, dtype=np.int64)
    valid = np.all((tri >= 0) & (tri < int(num_nodes)), axis=1)
    return tri[valid]


def _cells_to_tetrahedra_numpy(cells: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Extract linear tetrahedra from common tetra/hex volume cell arrays."""
    arr = cells.detach().to(device="cpu", dtype=torch.long).numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    nv = int(arr.shape[1])
    if nv == 4:
        tet = arr
    elif nv == 10:
        # Quadratic tetrahedra conventionally store the four vertices first.
        tet = arr[:, :4]
    elif nv == 8:
        # Standard hexahedron split into five tetrahedra. This assumes the usual
        # corner ordering; unsupported orderings should be converted upstream.
        tet = np.concatenate(
            [
                arr[:, [0, 1, 3, 4]],
                arr[:, [1, 2, 3, 6]],
                arr[:, [1, 3, 4, 6]],
                arr[:, [1, 5, 6, 4]],
                arr[:, [3, 6, 7, 4]],
            ],
            axis=0,
        )
    else:
        return np.empty((0, 4), dtype=np.int64)
    valid = np.all((tet >= 0) & (tet < int(num_nodes)), axis=1)
    return tet[valid]


def _select_fem_vectors(
    solver_vectors: torch.Tensor,
    inv_sqrt_mass: torch.Tensor,
    kind: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if kind == "v":
        return solver_vectors
    if kind == "u":
        return inv_sqrt_mass[:, None] * solver_vectors
    if kind == "both":
        return {
            "u": inv_sqrt_mass[:, None] * solver_vectors,
            "v": solver_vectors,
        }
    raise ValueError(f"Unsupported FEM eigenvector kind {kind!r}; expected 'u' or 'v'.")


def _fem3d_laplacian_eigenpairs_one(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = int(pos.shape[0])
    k = int(num_eigenvectors)
    device = pos.device
    if num_nodes == 0:
        return torch.empty((k,), device=device), torch.empty((0, k), device=device)
    if cells is None or cells.numel() == 0:
        raise ValueError("FEM3D Laplacian requires volume cells.")

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:  # pragma: no cover
        sp = None
        spla = None

    pos_np = pos.detach().to(device="cpu", dtype=torch.float64).numpy()
    tet = _cells_to_tetrahedra_numpy(cells, num_nodes)
    if tet.size == 0:
        raise ValueError("FEM3D Laplacian requires tetrahedral or supported hexahedral volume cells.")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    mass = np.zeros((num_nodes,), dtype=np.float64)
    ref_grads = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    for cell in tet:
        verts = pos_np[cell]
        jac = np.stack([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]], axis=1)
        det = float(np.linalg.det(jac))
        volume = abs(det) / 6.0
        if volume <= 1e-14:
            continue
        try:
            inv_jac = np.linalg.inv(jac)
        except np.linalg.LinAlgError:
            continue
        grads = ref_grads @ inv_jac
        local_k = volume * (grads @ grads.T)
        rr = np.repeat(cell, 4)
        cc = np.tile(cell, 4)
        rows.append(rr)
        cols.append(cc)
        data.append(local_k.reshape(-1))
        np.add.at(mass, cell, volume / 4.0)

    if not data:
        raise ValueError("FEM3D Laplacian found no nondegenerate volume cells.")

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(data)
    inv_sqrt_mass = 1.0 / np.sqrt(np.maximum(mass, 1e-12))
    if sp is not None and spla is not None and num_nodes > k + 2:
        stiffness = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
        operator = sp.diags(inv_sqrt_mass) @ stiffness @ sp.diags(inv_sqrt_mass)
        operator = 0.5 * (operator + operator.T)
        try:
            eigenvalues_np, eigenvectors_np = spla.eigsh(
                operator,
                k=min(k + 1, num_nodes - 1),
                which="SM",
                tol=1e-3,
                maxiter=2000,
            )
        except Exception:
            eigenvalues_np, eigenvectors_np = np.linalg.eigh(operator.toarray())
        eigvals = torch.from_numpy(np.asarray(eigenvalues_np, dtype=np.float32)).to(device=device)
        eigvecs = torch.from_numpy(np.asarray(eigenvectors_np, dtype=np.float32)).to(device=device)
    else:
        stiffness = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
        row_t = torch.from_numpy(row.astype(np.int64)).to(device=device)
        col_t = torch.from_numpy(col.astype(np.int64)).to(device=device)
        val_t = torch.from_numpy(val.astype(np.float32)).to(device=device)
        stiffness.index_put_((row_t, col_t), val_t, accumulate=True)
        inv_sqrt_mass_t_dense = torch.from_numpy(inv_sqrt_mass.astype(np.float32)).to(device=device)
        operator = inv_sqrt_mass_t_dense[:, None] * stiffness * inv_sqrt_mass_t_dense[None, :]
        operator = 0.5 * (operator + operator.t())
        eigvals, eigvecs = torch.linalg.eigh(operator)

    vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigvecs.unsqueeze(0), k)
    inv_sqrt_mass_t = torch.from_numpy(inv_sqrt_mass.astype(np.float32)).to(device=device)
    return vals[0], _select_fem_vectors(vecs[0], inv_sqrt_mass_t, vector_kind)


def _fem3d_laplacian_eigenvectors_one(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    del init_eigenvectors
    return _fem3d_laplacian_eigenpairs_one(
        pos,
        cells,
        num_eigenvectors,
        vector_kind=vector_kind,
    )[1]


def _fem_laplacian_eigenvectors_one(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    num_nodes = int(pos.shape[0])
    k = int(num_eigenvectors)
    device = pos.device
    if num_nodes == 0:
        return torch.empty((0, k), device=device)
    if cells is None or cells.numel() == 0:
        return torch.zeros((num_nodes, k), device=device, dtype=torch.float32)

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:  # pragma: no cover
        sp = None
        spla = None

    if sp is not None and spla is not None and num_nodes > k + 2:
        pos_np = pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        tri = _cells_to_triangles_numpy(cells, num_nodes)
        if tri.size == 0:
            return torch.zeros((num_nodes, k), device=device, dtype=torch.float32)

        ia, ib, ic = tri[:, 0], tri[:, 1], tri[:, 2]
        pa, pb, pc = pos_np[ia], pos_np[ib], pos_np[ic]
        cross = np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))
        valid = cross > 1e-12
        ia, ib, ic = ia[valid], ib[valid], ic[valid]
        pa, pb, pc = pa[valid], pb[valid], pc[valid]
        cross = cross[valid]
        if cross.size == 0:
            return torch.zeros((num_nodes, k), device=device, dtype=torch.float32)

        area = 0.5 * cross
        cot_a = np.einsum("ij,ij->i", pb - pa, pc - pa) / cross
        cot_b = np.einsum("ij,ij->i", pa - pb, pc - pb) / cross
        cot_c = np.einsum("ij,ij->i", pa - pc, pb - pc) / cross

        rows = []
        cols = []
        data = []
        for i, j, weight in ((ib, ic, 0.5 * cot_a), (ia, ic, 0.5 * cot_b), (ia, ib, 0.5 * cot_c)):
            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            data.extend([weight, weight, -weight, -weight])
        mass = np.zeros((num_nodes,), dtype=np.float64)
        share = area / 3.0
        np.add.at(mass, ia, share)
        np.add.at(mass, ib, share)
        np.add.at(mass, ic, share)
        if device.type == "cuda" and torch.cuda.is_available():
            inv_sqrt_mass_t = torch.from_numpy((1.0 / np.sqrt(np.maximum(mass, 1e-8))).astype(np.float32)).to(device=device)
            row_t = torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(device=device)
            col_t = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(device=device)
            data_t = torch.from_numpy(np.concatenate(data).astype(np.float32)).to(device=device)
            stiffness = torch.sparse_coo_tensor(
                torch.stack([row_t, col_t]),
                data_t,
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
            idx = stiffness.indices()
            op_vals = inv_sqrt_mass_t[idx[0]] * stiffness.values() * inv_sqrt_mass_t[idx[1]]
            operator = torch.sparse_coo_tensor(idx, op_vals, (num_nodes, num_nodes), device=device).coalesce()
            try:
                _, eigenvectors = _lobpcg_ladder(
                    operator,
                    k,
                    ladder=(16, k + 1),
                    niter=60,
                    tol=1e-4,
                    init=init_eigenvectors,
                )
                out = _select_nonconstant_eigenvectors(eigenvectors.unsqueeze(0), k).squeeze(0)
                return _select_fem_vectors(out, inv_sqrt_mass_t, vector_kind)
            except Exception:
                pass

        stiffness = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        inv_sqrt_mass = 1.0 / np.sqrt(np.maximum(mass, 1e-8))
        operator = sp.diags(inv_sqrt_mass) @ stiffness @ sp.diags(inv_sqrt_mass)
        operator = 0.5 * (operator + operator.T)
        try:
            _, eigenvectors_np = spla.eigsh(operator, k=min(k + 1, num_nodes - 1), which="SM", tol=1e-3, maxiter=2000)
        except Exception:
            eigenvectors_np = np.linalg.eigh(operator.toarray())[1]

        eigenvectors = torch.from_numpy(np.asarray(eigenvectors_np, dtype=np.float32)).to(device=device)
        out = _select_nonconstant_eigenvectors(eigenvectors.unsqueeze(0), k).squeeze(0)
        inv_sqrt_mass_t = torch.from_numpy(inv_sqrt_mass.astype(np.float32)).to(device=device)
        return _select_fem_vectors(out, inv_sqrt_mass_t, vector_kind)

    pos = pos.to(device=device, dtype=torch.float32)
    stiffness = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    mass = torch.zeros((num_nodes,), device=device, dtype=torch.float32)

    def add_pair(i: int, j: int, weight: torch.Tensor) -> None:
        stiffness[i, i] += weight
        stiffness[j, j] += weight
        stiffness[i, j] -= weight
        stiffness[j, i] -= weight

    for cell in cells.to(device="cpu", dtype=torch.long):
        for ia, ib, ic in _triangulate_cell(cell):
            if ia < 0 or ib < 0 or ic < 0 or ia >= num_nodes or ib >= num_nodes or ic >= num_nodes:
                continue
            pa, pb, pc = pos[ia], pos[ib], pos[ic]
            cross = torch.abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])).clamp_min(1e-12)
            area = 0.5 * cross
            if float(area.item()) <= 1e-12:
                continue
            cot_a = torch.dot(pb - pa, pc - pa) / cross
            cot_b = torch.dot(pa - pb, pc - pb) / cross
            cot_c = torch.dot(pa - pc, pb - pc) / cross
            add_pair(ib, ic, 0.5 * cot_a)
            add_pair(ia, ic, 0.5 * cot_b)
            add_pair(ia, ib, 0.5 * cot_c)
            share = area / 3.0
            mass[ia] += share
            mass[ib] += share
            mass[ic] += share

    mass = mass.clamp_min(1e-8)
    inv_sqrt_mass = torch.rsqrt(mass)
    operator = inv_sqrt_mass[:, None] * stiffness * inv_sqrt_mass[None, :]
    operator = 0.5 * (operator + operator.t())
    _, eigenvectors = torch.linalg.eigh(operator)
    out = _select_nonconstant_eigenvectors(eigenvectors.unsqueeze(0), k).squeeze(0)
    return _select_fem_vectors(out, inv_sqrt_mass, vector_kind)


def _fem_laplacian_eigenpairs_one(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = int(pos.shape[0])
    k = int(num_eigenvectors)
    device = pos.device
    if num_nodes == 0:
        return torch.empty((k,), device=device), torch.empty((0, k), device=device)
    if cells is None or cells.numel() == 0:
        return (
            torch.zeros((k,), device=device, dtype=torch.float32),
            torch.zeros((num_nodes, k), device=device, dtype=torch.float32),
        )

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:  # pragma: no cover
        sp = None
        spla = None

    if sp is not None and spla is not None and num_nodes > k + 2:
        pos_np = pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        tri = _cells_to_triangles_numpy(cells, num_nodes)
        if tri.size == 0:
            return (
                torch.zeros((k,), device=device, dtype=torch.float32),
                torch.zeros((num_nodes, k), device=device, dtype=torch.float32),
            )

        ia, ib, ic = tri[:, 0], tri[:, 1], tri[:, 2]
        pa, pb, pc = pos_np[ia], pos_np[ib], pos_np[ic]
        cross = np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))
        valid = cross > 1e-12
        ia, ib, ic = ia[valid], ib[valid], ic[valid]
        pa, pb, pc = pa[valid], pb[valid], pc[valid]
        cross = cross[valid]
        if cross.size == 0:
            return (
                torch.zeros((k,), device=device, dtype=torch.float32),
                torch.zeros((num_nodes, k), device=device, dtype=torch.float32),
            )

        area = 0.5 * cross
        cot_a = np.einsum("ij,ij->i", pb - pa, pc - pa) / cross
        cot_b = np.einsum("ij,ij->i", pa - pb, pc - pb) / cross
        cot_c = np.einsum("ij,ij->i", pa - pc, pb - pc) / cross

        rows = []
        cols = []
        data = []
        for i, j, weight in ((ib, ic, 0.5 * cot_a), (ia, ic, 0.5 * cot_b), (ia, ib, 0.5 * cot_c)):
            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            data.extend([weight, weight, -weight, -weight])
        mass = np.zeros((num_nodes,), dtype=np.float64)
        share = area / 3.0
        np.add.at(mass, ia, share)
        np.add.at(mass, ib, share)
        np.add.at(mass, ic, share)
        if device.type == "cuda" and torch.cuda.is_available():
            inv_sqrt_mass_t = torch.from_numpy((1.0 / np.sqrt(np.maximum(mass, 1e-8))).astype(np.float32)).to(device=device)
            row_t = torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(device=device)
            col_t = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(device=device)
            data_t = torch.from_numpy(np.concatenate(data).astype(np.float32)).to(device=device)
            stiffness = torch.sparse_coo_tensor(
                torch.stack([row_t, col_t]),
                data_t,
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
            idx = stiffness.indices()
            op_vals = inv_sqrt_mass_t[idx[0]] * stiffness.values() * inv_sqrt_mass_t[idx[1]]
            operator = torch.sparse_coo_tensor(idx, op_vals, (num_nodes, num_nodes), device=device).coalesce()
            try:
                eigvals, eigenvectors = _lobpcg_ladder(
                    operator,
                    k,
                    ladder=(16, k + 1),
                    niter=60,
                    tol=1e-4,
                    init=init_eigenvectors,
                )
                vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigenvectors.unsqueeze(0), k)
                return vals[0], _select_fem_vectors(vecs[0], inv_sqrt_mass_t, vector_kind)
            except Exception:
                pass

        stiffness = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        inv_sqrt_mass = 1.0 / np.sqrt(np.maximum(mass, 1e-8))
        operator = sp.diags(inv_sqrt_mass) @ stiffness @ sp.diags(inv_sqrt_mass)
        operator = 0.5 * (operator + operator.T)
        try:
            eigenvalues_np, eigenvectors_np = spla.eigsh(
                operator,
                k=min(k + 1, num_nodes - 1),
                which="SM",
                tol=1e-3,
                maxiter=2000,
            )
        except Exception:
            eigenvalues_np, eigenvectors_np = np.linalg.eigh(operator.toarray())

        eigvals = torch.from_numpy(np.asarray(eigenvalues_np, dtype=np.float32)).to(device=device)
        eigvecs = torch.from_numpy(np.asarray(eigenvectors_np, dtype=np.float32)).to(device=device)
        vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigvecs.unsqueeze(0), k)
        inv_sqrt_mass_t = torch.from_numpy(inv_sqrt_mass.astype(np.float32)).to(device=device)
        return vals[0], _select_fem_vectors(vecs[0], inv_sqrt_mass_t, vector_kind)

    pos = pos.to(device=device, dtype=torch.float32)
    stiffness = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    mass = torch.zeros((num_nodes,), device=device, dtype=torch.float32)

    def add_pair(i: int, j: int, weight: torch.Tensor) -> None:
        stiffness[i, i] += weight
        stiffness[j, j] += weight
        stiffness[i, j] -= weight
        stiffness[j, i] -= weight

    for cell in cells.to(device="cpu", dtype=torch.long):
        for ia, ib, ic in _triangulate_cell(cell):
            if ia < 0 or ib < 0 or ic < 0 or ia >= num_nodes or ib >= num_nodes or ic >= num_nodes:
                continue
            pa, pb, pc = pos[ia], pos[ib], pos[ic]
            cross = torch.abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])).clamp_min(1e-12)
            area = 0.5 * cross
            if float(area.item()) <= 1e-12:
                continue
            cot_a = torch.dot(pb - pa, pc - pa) / cross
            cot_b = torch.dot(pa - pb, pc - pb) / cross
            cot_c = torch.dot(pa - pc, pb - pc) / cross
            add_pair(ib, ic, 0.5 * cot_a)
            add_pair(ia, ic, 0.5 * cot_b)
            add_pair(ia, ib, 0.5 * cot_c)
            share = area / 3.0
            mass[ia] += share
            mass[ib] += share
            mass[ic] += share

    mass = mass.clamp_min(1e-8)
    inv_sqrt_mass = torch.rsqrt(mass)
    operator = inv_sqrt_mass[:, None] * stiffness * inv_sqrt_mass[None, :]
    operator = 0.5 * (operator + operator.t())
    eigvals, eigvecs = torch.linalg.eigh(operator)
    vals, vecs = _select_nonconstant_eigenpairs(eigvals.unsqueeze(0), eigvecs.unsqueeze(0), k)
    return vals[0], _select_fem_vectors(vecs[0], inv_sqrt_mass, vector_kind)


def _batched_fem_laplacian_eigenvectors(
    positions: list[torch.Tensor],
    cells: list[torch.Tensor | None],
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    features = [
        _fem_laplacian_eigenvectors_one(
            pos.to(dtype=torch.float32),
            cell,
            num_eigenvectors,
            vector_kind=vector_kind,
            init_eigenvectors=(
                init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None
            ),
        )
        for i, (pos, cell) in enumerate(zip(positions, cells, strict=True))
    ]
    return torch.stack(features, dim=0)


def _batched_fem_laplacian_eigenpairs(
    positions: list[torch.Tensor],
    cells: list[torch.Tensor | None],
    num_eigenvectors: int,
    *,
    vector_kind: str,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        _fem_laplacian_eigenpairs_one(
            pos.to(dtype=torch.float32),
            cell,
            num_eigenvectors,
            vector_kind=vector_kind,
            init_eigenvectors=(
                init_eigenvectors[i] if init_eigenvectors is not None and init_eigenvectors.shape[0] > i else None
            ),
        )
        for i, (pos, cell) in enumerate(zip(positions, cells, strict=True))
    ]
    values, vectors = zip(*pairs, strict=True)
    return torch.stack(list(values), dim=0), torch.stack(list(vectors), dim=0)


def _fem_laplacian_eigenpairs_both_one(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    num_eigenvectors: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = pos.to(dtype=torch.float32)
    if int(pos.shape[-1]) == 2:
        values, vectors = _fem_laplacian_eigenpairs_one(
            pos,
            cells,
            num_eigenvectors,
            vector_kind="both",
            init_eigenvectors=init_eigenvectors,
        )
    elif int(pos.shape[-1]) == 3:
        values, vectors = _fem3d_laplacian_eigenpairs_one(
            pos,
            cells,
            num_eigenvectors,
            vector_kind="both",
            init_eigenvectors=init_eigenvectors,
        )
    else:
        raise ValueError(f"FEM Laplacian requires 2D planar or 3D volume coordinates; got pos dim={int(pos.shape[-1])}.")
    if not isinstance(vectors, dict):
        raise RuntimeError("Internal FEM both-mode expected a vector dictionary.")
    return values, vectors["u"], vectors["v"]


def compute_laplacian_fem_eigendecomp_both(
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    count: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fem_laplacian_eigenpairs_both_one(
        pos.to(dtype=torch.float32),
        cells,
        int(count),
        init_eigenvectors=init_eigenvectors,
    )


def compute_laplacian_features(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    spec: str | None,
    default_dim: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = []
    prev_init = init_eigenvectors
    for name, count in parse_laplacian_spec(spec, default_dim):
        part = compute_laplacian_feature_part(
            edge_index,
            pos,
            cells,
            name,
            count,
            init_eigenvectors=prev_init,
        )
        parts.append(part)
        prev_init = part
    return torch.cat(parts, dim=-1)


def compute_laplacian_eigendecomp(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    spec: str | None,
    default_dim: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    vectors = []
    prev_init = init_eigenvectors
    for name, count in parse_laplacian_spec(spec, default_dim):
        eigvals, eigvecs = compute_laplacian_eigendecomp_part(
            edge_index,
            pos,
            cells,
            name,
            count,
            init_eigenvectors=prev_init,
        )
        values.append(eigvals)
        vectors.append(eigvecs)
        prev_init = eigvecs
    return torch.cat(values, dim=-1), torch.cat(vectors, dim=-1)


def compute_laplacian_feature_part(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    name: str,
    count: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
    if name == "graph":
        num_nodes = int(pos.shape[0])
        return _batched_laplacian_eigenvectors(
            num_nodes,
            [edge_index],
            count,
            init_eigenvectors=init_eigenvectors.unsqueeze(0) if init_eigenvectors is not None else None,
        )[0]
    if name == "edge":
        num_nodes = int(pos.shape[0])
        return _batched_edge_weighted_laplacian_eigenvectors(
            num_nodes,
            [edge_index],
            [pos],
            count,
            init_eigenvectors=init_eigenvectors.unsqueeze(0) if init_eigenvectors is not None else None,
        )[0]
    if name in {"fem_u", "fem_v"}:
        values, vectors_u, vectors_v = _fem_laplacian_eigenpairs_both_one(
            pos.to(device=edge_index.device, dtype=torch.float32),
            cells,
            count,
            init_eigenvectors=init_eigenvectors,
        )
        del values
        return vectors_u if name == "fem_u" else vectors_v
    raise AssertionError(name)


def compute_laplacian_eigendecomp_part(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor | None,
    name: str,
    count: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "graph":
        num_nodes = int(pos.shape[0])
        values, vectors = _batched_laplacian_eigenpairs(
            num_nodes,
            [edge_index],
            count,
            init_eigenvectors=init_eigenvectors.unsqueeze(0) if init_eigenvectors is not None else None,
        )
        return values[0], vectors[0]
    if name == "edge":
        num_nodes = int(pos.shape[0])
        values, vectors = _batched_edge_weighted_laplacian_eigenpairs(
            num_nodes,
            [edge_index],
            [pos],
            count,
            init_eigenvectors=init_eigenvectors.unsqueeze(0) if init_eigenvectors is not None else None,
        )
        return values[0], vectors[0]
    if name in {"fem_u", "fem_v"}:
        values, vectors_u, vectors_v = _fem_laplacian_eigenpairs_both_one(
            pos.to(device=edge_index.device, dtype=torch.float32),
            cells,
            count,
            init_eigenvectors=init_eigenvectors,
        )
        return values, vectors_u if name == "fem_u" else vectors_v
    raise AssertionError(name)

