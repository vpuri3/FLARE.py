from pdebench.config import _coerce_model_config
from pdebench.models.graph_models.glt import (
    GeoTransolverPEConfig,
    MultiscaleHopPEConfig,
    NonePEConfig,
    ProbeDistPEConfig,
    RawEigenPEConfig,
    SpectralFilterPEConfig,
)


def test_coerce_glt_pe_raw_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe_inject_mode": "concat_qk",
        "pe_update": False,
        "pe": {"kind": "raw_eigen", "num_eigenmodes": 64, "laplacian_spec": "graph"},
    })
    assert cfg.pe_inject_mode == "concat_qk"
    assert isinstance(cfg.pe, RawEigenPEConfig)
    assert cfg.pe.num_eigenmodes == 64
    fr = cfg.pe.to_feature_request()
    assert fr.laplacian_k == 64


def test_coerce_glt_spe_pe_config_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe": {"kind": "spe", "num_eigenmodes": 8, "filter_type": "spatial", "mode": "multihop"},
    })
    assert isinstance(cfg.pe, SpectralFilterPEConfig)
    assert cfg.pe.filter_type == "spatial"
    assert cfg.pe.mode == "multihop"


def test_coerce_glt_pe_raw_from_flattened_cli_keys():
    """jsonargparse flattens ``--model.pe.kind`` to a literal ``"pe.kind"`` dict key
    (Config.model is only typed as ``ModelConfig | dict`` at parse time); the coerce
    path must unflatten those dotted keys under ``pe`` before building ``GLTConfig``."""
    cfg = _coerce_model_config({
        "model": "glt",
        "pe_inject_mode": "concat_qk",
        "pe.kind": "raw_eigen",
        "pe.num_eigenmodes": 64,
        "pe.laplacian_spec": "graph",
    })
    assert cfg.pe_inject_mode == "concat_qk"
    assert isinstance(cfg.pe, RawEigenPEConfig)
    assert cfg.pe.num_eigenmodes == 64
    assert cfg.pe.laplacian_spec == "graph"


def test_coerce_glt_pe_spe_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe.kind": "spe",
        "pe.num_eigenmodes": 8,
        "pe.filter_type": "band",
        "pe.mode": "query",
    })
    assert isinstance(cfg.pe, SpectralFilterPEConfig)
    assert cfg.pe.num_eigenmodes == 8
    assert cfg.pe.filter_type == "band"
    assert cfg.pe.mode == "query"


def test_coerce_glt_none_pe_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe_inject_mode": "concat_input",
        "pe": {"kind": "none"},
    })
    assert isinstance(cfg.pe, NonePEConfig)
    assert cfg.pe.kind == "none"
    assert cfg.pe.to_feature_request().laplacian_k == 0


def test_coerce_glt_none_pe_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe.kind": "none",
    })
    assert isinstance(cfg.pe, NonePEConfig)


def test_coerce_glt_geo_transolver_pe_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe": {
            "kind": "geo_transolver_pe",
            "radii": [0.05, 0.25],
            "neighbors_in_radius": [8, 32],
            "n_hidden_local": 32,
        },
    })
    assert isinstance(cfg.pe, GeoTransolverPEConfig)
    assert cfg.pe.radii == (0.05, 0.25)
    assert cfg.pe.neighbors_in_radius == (8, 32)
    assert cfg.pe.n_hidden_local == 32
    assert cfg.pe.to_feature_request().laplacian_k == 0


def test_coerce_glt_geo_transolver_pe_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe.kind": "geo_transolver_pe",
        "pe.radii": [0.1],
        "pe.neighbors_in_radius": [4],
        "pe.n_hidden_local": 16,
    })
    assert isinstance(cfg.pe, GeoTransolverPEConfig)
    assert cfg.pe.radii == (0.1,)
    assert cfg.pe.neighbors_in_radius == (4,)
    assert cfg.pe.n_hidden_local == 16


def test_coerce_glt_multiscale_hop_pe_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe": {
            "kind": "multiscale_hop_pe",
            "pointnet_hidden_dim": 32,
            "normalize_by_mean_edge_length": False,
        },
    })
    assert isinstance(cfg.pe, MultiscaleHopPEConfig)
    assert cfg.pe.pointnet_hidden_dim == 32
    assert cfg.pe.normalize_by_mean_edge_length is False
    assert cfg.pe.to_feature_request().laplacian_k == 0


def test_coerce_glt_multiscale_hop_pe_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe.kind": "multiscale_hop_pe",
        "pe.pointnet_hidden_dim": 32,
        "pe.normalize_by_mean_edge_length": False,
    })
    assert isinstance(cfg.pe, MultiscaleHopPEConfig)
    assert cfg.pe.pointnet_hidden_dim == 32
    assert cfg.pe.normalize_by_mean_edge_length is False


def test_coerce_glt_probe_dist_pe_from_dict():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe": {
            "kind": "probe_dist",
            "num_probes": 16,
            "geodesic_feats": True,
            "num_anchor_candidates": 3,
            "max_geodesic_hops": 9,
            "temperature": 0.75,
            "distance_cap": 1.5,
            "init_eps": 1e-3,
            "distance_eps": 1e-10,
        },
    })
    assert isinstance(cfg.pe, ProbeDistPEConfig)
    assert cfg.pe.num_probes == 16
    assert cfg.pe.geodesic_feats is True
    assert cfg.pe.euclidean_feats is True
    assert cfg.pe.num_anchor_candidates == 3
    assert cfg.pe.to_feature_request().edges is True


def test_coerce_glt_probe_dist_pe_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "glt",
        "pe.kind": "probe_dist",
        "pe.num_probes": 8,
        "pe.geodesic_feats": False,
        "pe.init_eps": 1e-3,
    })
    assert isinstance(cfg.pe, ProbeDistPEConfig)
    assert cfg.pe.num_probes == 8
    assert cfg.pe.geodesic_feats is False
