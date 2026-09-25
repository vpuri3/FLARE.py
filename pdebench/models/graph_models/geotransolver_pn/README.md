# Vendored PhysicsNeMo GeoTransolver (GALE)

Irregular-mesh GALE stack adapted from NVIDIA PhysicsNeMo
`physicsnemo/experimental/models/geotransolver/` (Apache-2.0).

Upstream: https://github.com/NVIDIA/physicsnemo  
Vendored against `main` (fetched 2026-07-14).

## Layout

| File | Role |
|------|------|
| `ball_query.py` | Pure-PyTorch BQWarp stand-in (radius + pad/truncate to K) |
| `pn_compat.py` | `Mlp`, soft-slice helpers, irregular physics attention |
| `context_projector.py` | ContextProjector, multi-scale extractors, GlobalContextBuilder |
| `gale.py` | GALE + GALE_block (irregular only) |
| `geotransolver_core.py` | GeoTransolver core forward (irregular only) |

Public pdebench entrypoint remains `pdebench.models.graph_models.geo_transolver`
(`GeoTransolverConfig` / `GeoTransolverModel`), which packs GINOT flat graphs
into `(B, N, *)` and calls this core.
