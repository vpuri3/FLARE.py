import torch

from pdebench.models.graph_models.rigno import RIGNOConfig, RIGNOModel


def test_rigno_forward_pos_only_no_edges():
    cfg = RIGNOConfig(channel_dim=32, num_blocks=2, num_slices=4, region_knn=4)
    model = RIGNOModel(cfg, metadata={"c_in": 3, "c_out": 2})
    pos = torch.rand(20, 2)
    feats = torch.rand(20, 1)
    y = model(pos=pos, feats=feats)
    assert y.shape == (20, 2)


def test_rigno_does_not_import_meshgraphnets():
    import pdebench.models.graph_models.rigno as m

    assert not hasattr(m, "MeshGraphNetProcessor")
    src = open(m.__file__).read()
    assert "meshgraphnets" not in src


def test_geo_transolver_ball_query_no_edges():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    cfg = GeoTransolverConfig(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_slices=8,
        include_local_features=True,
        ball_radii=(0.2, 0.5),
        ball_ks=(4, 8),
        n_hidden_local=8,
    )
    model = GeoTransolverModel(cfg, metadata={"c_in": 3, "c_out": 1, "space_dim": 2})
    y = model(pos=torch.rand(16, 2), feats=torch.rand(16, 1))
    assert y.shape == (16, 1)


def test_geo_transolver_no_edge_path():
    """GALE path never requires mesh edges."""
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    cfg = GeoTransolverConfig(channel_dim=32, num_blocks=1, num_heads=4, num_slices=4, n_hidden_local=8)
    model = GeoTransolverModel(cfg, metadata={"c_in": 3, "c_out": 1, "space_dim": 2})
    y = model(pos=torch.rand(8, 2), feats=torch.rand(8, 1))
    assert y.shape == (8, 1)


def test_geo_transolver_can_skip_local_feature_concatenation():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    assert GeoTransolverConfig().concat_local_features is True
    cfg = GeoTransolverConfig(
        channel_dim=32,
        num_blocks=1,
        num_heads=4,
        num_slices=8,
        include_local_features=True,
        concat_local_features=False,
        ball_radii=(0.05, 0.25),
        ball_ks=(4, 8),
        n_hidden_local=8,
    )
    model = GeoTransolverModel(cfg, metadata={"c_in": 3, "c_out": 2, "space_dim": 2})

    assert model.core.effective_hidden == 32
    assert model.core.context_builder.local_extractors is not None
    assert model.core.context_builder.get_context_dim() == 24

    y = model(pos=torch.rand(12, 2), feats=torch.rand(12, 1))
    assert y.shape == (12, 2)


def test_ball_query_pad_to_k_handcrafted():
    from pdebench.models.graph_models.geotransolver_pn.ball_query import BQWarp

    # Two query points; only one neighbor within radius for q0, two for q1.
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    pts = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    bq = BQWarp(radius=0.1, neighbors_in_radius=3)
    mapping, neighbors = bq(x, pts)
    assert mapping.shape == (1, 2, 3)
    assert neighbors.shape == (1, 2, 3, 3)
    # q0 should see pts[0] and pts[1]; third slot padded with zeros / -1 index.
    assert int((mapping[0, 0] >= 0).sum()) == 2
    assert neighbors[0, 0, 2].abs().sum() == 0
    # q1 only pts[2] within radius 0.1 of (1,0,0)
    assert int((mapping[0, 1] >= 0).sum()) == 1


def test_gale_context_dim_and_local_width():
    from pdebench.models.graph_models.geotransolver_pn import GeoTransolverCore

    radii = [0.05, 0.25]
    core = GeoTransolverCore(
        functional_dim=4,
        out_dim=2,
        geometry_dim=3,
        global_dim=3,
        n_layers=1,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        include_local_features=True,
        radii=radii,
        neighbors_in_radius=[4, 8],
        n_hidden_local=8,
    )
    # context: per-scale local (dim_head * |radii|) + geometry + global
    dim_head = 32 // 4
    expected_ctx = dim_head * len(radii) + dim_head + dim_head
    assert core.context_builder.get_context_dim() == expected_ctx
    assert core.effective_hidden == 32 + 8 * len(radii)

    # GALE scalar mix present on attention module
    gale = core.blocks[0].Attn
    assert hasattr(gale, "state_mixing")
    assert gale.state_mixing_mode == "weighted"

    b, n = 2, 12
    local = torch.randn(b, n, 4)
    pos = torch.rand(b, n, 3)
    glob = torch.randn(b, 1, 3)
    out = core(local, local_positions=pos, geometry=pos, global_embedding=glob)
    assert out.shape == (b, n, 2)


def _gale_cross_attrs(gale):
    return ("cross_q", "cross_k", "cross_v", "state_mixing", "concat_project")


def _no_usable_context_builder(core) -> bool:
    return not hasattr(core, "context_builder") or core.context_builder is None


def test_geo_transolver_config_use_geo_defaults_true():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig

    assert GeoTransolverConfig().use_geo is True


def test_geo_transolver_use_geo_false_disables_geo_path():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    cfg = GeoTransolverConfig(
        use_geo=False,
        channel_dim=32,
        num_blocks=1,
        num_heads=4,
        num_slices=8,
        include_local_features=True,
        concat_local_features=True,
        ball_radii=(0.05, 0.25),
        ball_ks=(4, 8),
        n_hidden_local=8,
    )
    model = GeoTransolverModel(cfg, metadata={"c_in": 3, "c_out": 2, "space_dim": 2})
    core = model.core

    assert _no_usable_context_builder(core)
    assert core.effective_hidden == 32

    gale = core.blocks[0].Attn
    for attr in _gale_cross_attrs(gale):
        assert not hasattr(gale, attr), f"expected no {attr} when use_geo=False"

    y = model(pos=torch.rand(12, 2), feats=torch.rand(12, 1))
    assert y.shape == (12, 2)


def test_geo_transolver_use_geo_true_preserves_local_features():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    cfg = GeoTransolverConfig(
        channel_dim=32,
        num_blocks=1,
        num_heads=4,
        num_slices=8,
        include_local_features=True,
        concat_local_features=True,
        ball_radii=(0.05, 0.25),
        ball_ks=(4, 8),
        n_hidden_local=8,
    )
    assert cfg.use_geo is True
    model = GeoTransolverModel(cfg, metadata={"c_in": 3, "c_out": 2, "space_dim": 2})
    core = model.core

    assert not _no_usable_context_builder(core)
    assert core.context_builder.local_extractors is not None
    assert core.effective_hidden == 32 + 8 * len(cfg.ball_radii)

    gale = core.blocks[0].Attn
    assert hasattr(gale, "cross_q")
    assert hasattr(gale, "state_mixing")

    y = model(pos=torch.rand(12, 2), feats=torch.rand(12, 1))
    assert y.shape == (12, 2)


def test_geo_transolver_multiscale_bumper_smoke():
    from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig, GeoTransolverModel

    cfg = GeoTransolverConfig(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_slices=8,
        include_local_features=True,
        geometry_dim=3,
        global_dim=3,
        ball_radii=(0.05, 0.25),
        ball_ks=(4, 8),
        n_hidden_local=8,
    )
    # bumper-like: space 3 + thickness 1 + globals 3
    model = GeoTransolverModel(cfg, metadata={"c_in": 7, "c_out": 3, "space_dim": 3})
    n0, n1 = 10, 14
    pos = torch.cat([torch.rand(n0, 3), torch.rand(n1, 3)], dim=0)
    feats = torch.cat([torch.rand(n0, 4), torch.rand(n1, 4)], dim=0)
    # make globals constant per sample
    feats[:n0, 1:] = torch.tensor([0.1, 0.2, 0.3])
    feats[n0:, 1:] = torch.tensor([0.4, 0.5, 0.6])
    batch_index = torch.cat(
        [torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)],
        dim=0,
    )
    y = model(pos=pos, feats=feats, batch_index=batch_index)
    assert y.shape == (n0 + n1, 3)
