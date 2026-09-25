from __future__ import annotations

import inspect

import pytest
from jsonargparse import ArgumentParser

import pdebench
import pdebench.models.model_factory as model_factory
from pdebench import __main__ as pdebench_main
from pdebench.config import (
    Config,
    DatasetConfig,
    FlareConfig,
    FlarePPConfig,
    GinotDatasetConfig,
    MeshGraphNetConfig,
    ModelConfig,
    OneCycleSchedulerConfig,
    OptimizerConfig,
    PlateauSchedulerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
    TransolverConfig,
)
from pdebench.dataset.utils import compile_stats_model_for_dataset
from pdebench.models.graph_models.gito import GITOConfig


def make_cfg(
    *,
    use_puri2025flare_config: bool | None = None,
    run: dict | None = None,
    dataset: dict | None = None,
    training: dict | None = None,
    optimizer: dict | None = None,
    scheduler: dict | None = None,
    model: dict | None = None,
) -> Config:
    cfg = Config(
        run=RunConfig(**(run or {})),
        dataset=dataset or {},
        training=TrainingConfig(**(training or {})),
        optimizer=OptimizerConfig(**(optimizer or {})),
        scheduler=scheduler or {},
        model=model or {},
    )
    if use_puri2025flare_config is not None:
        cfg.use_puri2025flare_config = use_puri2025flare_config
    return cfg


def test_main_make_model_exports_factory_function() -> None:
    assert pdebench_main.make_model is model_factory.make_model


def test_ginot_varlen_stats_use_eager_model_when_training_is_compiled() -> None:
    assert compile_stats_model_for_dataset("micro_puc_fixed", "glt", compile_model=True) is False
    assert compile_stats_model_for_dataset("micro_puc_fixed", "glt", compile_model=False) is False
    assert compile_stats_model_for_dataset("elasticity", "flare", compile_model=True) is True
    assert compile_stats_model_for_dataset("elasticity", "flare", compile_model=False) is False
    assert compile_stats_model_for_dataset("lpbf", "flare", compile_model=True) is True
    assert compile_stats_model_for_dataset("lpbf", "glt", compile_model=True) is False
    assert compile_stats_model_for_dataset("drivaerml_surface", "flare", compile_model=True) is False
    assert compile_stats_model_for_dataset("drivaerml_surface", "flarepp", compile_model=True) is False
    assert compile_stats_model_for_dataset("drivaerml_surface", "flare", compile_model=False) is False


def test_public_model_constructors_take_config_and_metadata() -> None:
    for cls in (
        pdebench.FLAREModel,
        pdebench.FLAREPPModel,
        pdebench.TransformerWrapper,
        pdebench.Transolver,
        pdebench.LNO,
        pdebench.GNOT,
    ):
        params = list(inspect.signature(cls.__init__).parameters)
        assert params[:3] == ["self", "config", "metadata"]


def test_model_config_base_and_subclasses_remain_siblings() -> None:
    assert ModelConfig().model == "flare"
    assert FlareConfig.model == "flare"
    assert TransolverConfig.model == "transolver"
    assert hasattr(FlareConfig, "__dataclass_fields__")
    assert hasattr(TransolverConfig, "__dataclass_fields__")


def test_flarepp_config_registered_like_flare() -> None:
    assert FlarePPConfig.model == "flarepp"
    assert hasattr(FlarePPConfig, "__dataclass_fields__")
    assert "attn_scale" not in FlarePPConfig.__dataclass_fields__


def test_dataset_and_scheduler_config_bases_remain_shallow() -> None:
    assert DatasetConfig.__bases__ == (object,)
    assert GinotDatasetConfig.__bases__ == (DatasetConfig,)
    assert SchedulerConfig.__bases__ == (object,)
    assert OneCycleSchedulerConfig.__bases__ == (SchedulerConfig,)
    assert PlateauSchedulerConfig.__bases__ == (SchedulerConfig,)


def test_config_defaults_keep_scheduler_and_flare_family_values() -> None:
    cfg = Config()

    assert cfg.use_puri2025flare_config is False
    assert cfg.optimizer.optimizer == "adamw"
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert cfg.scheduler.pct_start == pytest.approx(0.10)
    assert cfg.scheduler.div_factor == pytest.approx(1e4)
    assert cfg.scheduler.final_div_factor == pytest.approx(1e4)
    assert cfg.training.compile_model is True
    assert cfg.training.ema is True
    assert cfg.model.num_layers_k_proj == 3
    assert cfg.model.num_layers_v_proj == 3
    assert cfg.model.num_layers_ffn == 3
    assert cfg.model.k_proj_mlp_ratio == pytest.approx(1.0)
    assert cfg.model.v_proj_mlp_ratio == pytest.approx(1.0)
    assert cfg.model.ffn_mlp_ratio == pytest.approx(1.0)
    assert cfg.model.qk_norm is False
    assert cfg.model.attn_scale == "one"

    serialized = cfg.to_dict()
    assert "num_layers_k_proj" in serialized["model"]
    assert "qk_norm" in serialized["model"]
    assert "graph_k_proj" not in serialized["model"]
    assert "x_k_layers" not in serialized["model"]
    assert "flare_glt_x_k_layers" not in serialized["model"]

    assert cfg.run.exp_name == "exp"
    assert cfg.dataset.dataset is None
    assert type(cfg.dataset) is DatasetConfig
    assert cfg.training.batch_size == 1
    assert cfg.optimizer.optimizer == "adamw"
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert isinstance(cfg.scheduler, OneCycleSchedulerConfig)
    assert isinstance(cfg.model, FlareConfig)
    assert cfg.model.num_layers_k_proj == 3
    assert cfg.model.qk_norm is False


def test_config_no_longer_exposes_flat_compatibility_attrs() -> None:
    cfg = Config()

    with pytest.raises(AttributeError):
        _ = cfg.batch_size
    with pytest.raises(AttributeError):
        _ = cfg.model_type


def test_cfg_nested_roundtrip_preserves_section_defaults() -> None:
    cfg = make_cfg(
        run={"train": True, "exp_name": "roundtrip"},
        dataset={"dataset": "elasticity"},
        training={"batch_size": 4, "epochs": 3},
        optimizer={"learning_rate": 5e-4},
        scheduler={"schedule": "ReduceLROnPlateau"},
        model={"model": "transolver"},
        use_puri2025flare_config=True,
    )

    restored = Config(**cfg.to_dict())

    assert restored.to_dict() == cfg.to_dict()
    assert restored.run.train is True
    assert restored.run.exp_name == "roundtrip"
    assert restored.dataset.dataset == "elasticity"
    assert type(restored.dataset).__name__ == "DatasetConfig"
    assert restored.training.batch_size == 4
    assert restored.optimizer.learning_rate == pytest.approx(5e-4)
    assert restored.scheduler.schedule == "ReduceLROnPlateau"
    assert type(restored.scheduler).__name__ == "PlateauSchedulerConfig"
    assert type(restored.model).__name__ == "TransolverConfig"


def test_cli_model_overrides_are_typed_after_dynamic_config_resolution() -> None:
    parser = ArgumentParser()
    parser.add_class_arguments(Config, nested_key=None)

    parsed = parser.parse_args(
        [
            "--dataset.dataset",
            "elasticity",
            "--model.model",
            "flare",
            "--model.channel_dim",
            "64",
            "--model.num_heads",
            "8",
            "--model.qk_norm",
            "true",
            "--training.steps",
            "10",
        ]
    )
    cfg = Config(**parsed.as_dict())

    assert cfg.model.channel_dim == 64
    assert isinstance(cfg.model.channel_dim, int)
    assert cfg.model.num_heads == 8
    assert isinstance(cfg.model.num_heads, int)
    assert cfg.model.qk_norm is True
    assert cfg.training.steps == 10
    assert isinstance(cfg.training.steps, int)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (["--run.not_a_field", "1"], "RunConfig.*not_a_field"),
        (["--dataset.not_a_field", "1"], "DatasetConfig.*not_a_field"),
        (["--dataset.dataset", "micro_puc", "--dataset.not_a_field", "1"], "GinotDatasetConfig.*not_a_field"),
        (["--training.not_a_field", "1"], "TrainingConfig.*not_a_field"),
        (["--optimizer.not_a_field", "1"], "OptimizerConfig.*not_a_field"),
        (["--scheduler.not_a_field", "1"], "OneCycleSchedulerConfig.*not_a_field"),
        (
            ["--scheduler.schedule", "ReduceLROnPlateau", "--scheduler.not_a_field", "1"],
            "PlateauSchedulerConfig.*not_a_field",
        ),
        (["--model.model", "flare", "--model.not_a_field", "1"], "FlareConfig.*not_a_field"),
        (["--model.model", "transolver", "--model.not_a_field", "1"], "TransolverConfig.*not_a_field"),
    ],
)
def test_cli_unknown_nested_config_fields_raise(args: list[str], match: str) -> None:
    parser = ArgumentParser()
    parser.add_class_arguments(Config, nested_key=None)
    parsed = parser.parse_args(args)

    with pytest.raises(ValueError, match=match):
        Config(**parsed.as_dict())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run": {"not_a_field": 1}}, "RunConfig.*not_a_field"),
        ({"dataset": {"not_a_field": 1}}, "DatasetConfig.*not_a_field"),
        ({"dataset": {"dataset": "micro_puc", "not_a_field": 1}}, "GinotDatasetConfig.*not_a_field"),
        ({"training": {"not_a_field": 1}}, "TrainingConfig.*not_a_field"),
        ({"optimizer": {"not_a_field": 1}}, "OptimizerConfig.*not_a_field"),
        ({"scheduler": {"not_a_field": 1}}, "OneCycleSchedulerConfig.*not_a_field"),
        (
            {"scheduler": {"schedule": "ReduceLROnPlateau", "not_a_field": 1}},
            "PlateauSchedulerConfig.*not_a_field",
        ),
        ({"model": {"model": "flare", "not_a_field": 1}}, "FlareConfig.*not_a_field"),
        ({"model": {"model": "transolver", "not_a_field": 1}}, "TransolverConfig.*not_a_field"),
    ],
)
def test_dict_unknown_nested_config_fields_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Config(**kwargs)


def test_training_config_owns_runtime_workers_and_context_parallel() -> None:
    cfg = Config(
        training=TrainingConfig(
            num_workers=6,
            prefetch_factor=3,
            use_context_parallel=True,
            context_parallel_size=4,
            cp_sequence_dim=2,
            cp_debug_gather_outputs=True,
        )
    )

    assert cfg.training.num_workers == 6
    assert cfg.training.prefetch_factor == 3
    assert cfg.training.use_context_parallel is True
    assert cfg.training.context_parallel_size == 4
    assert cfg.training.cp_sequence_dim == 2
    assert cfg.training.cp_debug_gather_outputs is True
    with pytest.raises(AttributeError):
        _ = cfg.use_context_parallel


def test_ginot_dataset_config_keeps_ginot_only_fields() -> None:
    cfg = Config(
        dataset={
            "dataset": "micro_puc",
            "mesh_split_seed": 11,
            "max_samples": 1024,
        }
    )

    assert isinstance(cfg.dataset, GinotDatasetConfig)
    assert cfg.dataset.mesh_split_seed == 11
    assert cfg.dataset.max_samples == 1024


def test_resolve_model_spec_meshgraphnet_defaults() -> None:
    cfg = make_cfg(
        dataset={"dataset": "tensile2d"},
        model={"model": "meshgraphnet", "channel_dim": 32, "num_blocks": 4},
        use_puri2025flare_config=True,
    )
    metadata = dict(c_in=6, c_out=3, c_edge=4)

    cfg_out, c_in, c_out, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert cfg_out is cfg
    assert c_in == 6
    assert c_out == 3
    assert model_name == "MeshGraphNetModel"
    assert model_ctor.__name__ == "MeshGraphNetModel"
    assert isinstance(cfg.model, MeshGraphNetConfig)
    assert cfg.model.channel_dim == 128
    assert cfg.model.num_blocks == 4
    assert cfg.training.batch_size == 1
    assert cfg.training.epochs == 100
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert cfg.training.compile_model is True
    assert cfg.training.ema is True


def test_resolve_model_spec_gito_puri_defaults() -> None:
    cfg = make_cfg(
        dataset={"dataset": "tensile2d"},
        model={"model": "gito", "channel_dim": 32},
        use_puri2025flare_config=True,
    )
    metadata = dict(c_in=6, c_out=3, c_edge=4)

    cfg_out, c_in, c_out, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert cfg_out is cfg
    assert c_in == 6
    assert c_out == 3
    assert model_name == "GITOModel"
    assert model_ctor.__name__ == "GITOModel"
    assert isinstance(cfg.model, GITOConfig)
    assert cfg.model.channel_dim == 128
    assert cfg.model.num_blocks_hgt == 2
    assert cfg.model.num_blocks_self_attn == 0
    assert cfg.model.act == "silu"


def test_make_model_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="does_not_exist"):
        make_cfg(dataset={"dataset": "elasticity"}, model={"model": "does_not_exist"})


def test_resolve_model_spec_transolver_elasticity_defaults() -> None:
    cfg = make_cfg(dataset={"dataset": "elasticity"}, model={"model": "transolver"}, use_puri2025flare_config=True)
    metadata = dict(c_in=2, c_out=1, space_dim=2, fun_dim=0, time_cond=False)

    cfg_out, c_in, c_out, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert cfg_out is cfg
    assert c_in == 2
    assert c_out == 1
    assert model_name == "Transolver"
    assert model_ctor.__name__ == "Transolver"
    assert cfg.training.batch_size == 1
    assert cfg.training.epochs == 500
    assert cfg.scheduler.schedule == "CosineAnnealingLR"
    assert cfg.optimizer.learning_rate == pytest.approx(1e-3)
    assert cfg.optimizer.weight_decay == pytest.approx(1e-5)
    assert cfg.training.clip_grad_norm == pytest.approx(0.1)
    assert cfg.model.conv2d is False
    assert cfg.model.unified_pos is False
    assert cfg.training.compile_model is False
    assert cfg.training.ema is False
    assert cfg.training.mixed_precision is False
    assert cfg.model.num_blocks == 8
    assert cfg.model.channel_dim == 128
    assert cfg.model.num_slices == 64
    assert cfg.model.num_heads == 8


def test_resolve_model_spec_transolver_navier_stokes_defaults() -> None:
    cfg = make_cfg(
        dataset={"dataset": "navier_stokes"},
        model={"model": "transolver", "conv2d": True},
        use_puri2025flare_config=True,
    )
    metadata = dict(c_in=3, c_out=1, space_dim=2, fun_dim=0, time_cond=False, H=64, W=64)

    cfg_out, c_in, c_out, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert cfg_out is cfg
    assert c_in == 3
    assert c_out == 1
    assert model_name == "Transolver_Structured_Mesh_2D"
    assert model_ctor.__name__ == "Transolver_Structured_Mesh_2D"
    assert cfg.training.batch_size == 2
    assert cfg.training.epochs == 500
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert cfg.training.clip_grad_norm is None
    assert cfg.model.conv2d is True
    assert cfg.model.unified_pos is True
    assert cfg.training.compile_model is False
    assert cfg.training.ema is False
    assert cfg.training.mixed_precision is False
    assert cfg.model.num_blocks == 8
    assert cfg.model.channel_dim == 256
    assert cfg.model.num_slices == 32
    assert cfg.model.num_heads == 8


def test_resolve_model_spec_transolver_conv2d_rejected_on_elasticity() -> None:
    cfg = make_cfg(dataset={"dataset": "elasticity"}, model={"model": "transolver", "conv2d": True})
    metadata = dict(c_in=2, c_out=1, space_dim=2, fun_dim=0)

    with pytest.raises(ValueError, match="model.conv2d=True"):
        model_factory._resolve_model_spec(cfg, metadata)


def test_resolve_model_spec_flare_attn_scale_auto_sqrt_for_wide_heads() -> None:
    cfg = make_cfg(
        dataset={"dataset": "tensile2d"},
        model={"model": "flare", "channel_dim": 256, "num_heads": 8, "attn_scale": "one"},
    )
    metadata = dict(c_in=6, c_out=2, space_dim=2, fun_dim=0, time_cond=False)

    cfg_out, _, _, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert cfg_out is cfg
    assert model_name == "FLARE"
    assert model_ctor.__name__ == "FLAREModel"
    assert cfg.model.attn_scale == pytest.approx((32.0) ** -0.5)


def test_resolve_model_spec_glt_allows_lpbf() -> None:
    from pdebench.dataset.lpbf import LPBF_DATASETS

    assert LPBF_DATASETS <= frozenset(model_factory.STATIC_MESH_DATASETS)

    cfg = make_cfg(
        dataset={"dataset": "lpbf"},
        model={"model": "glt", "channel_dim": 128, "num_blocks": 4, "num_heads": 8},
    )
    metadata = dict(c_in=3, c_out=4, space_dim=3, fun_dim=0, time_cond=False)

    _, _, _, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)

    assert model_name == "GLT"
    assert model_ctor is pdebench.GLT


def test_resolve_model_spec_rejects_removed_ginot() -> None:
    with pytest.raises(ValueError, match="ginot"):
        make_cfg(dataset={"dataset": "micro_puc"}, model={"model": "ginot"})


def test_resolve_model_spec_graph_flare_is_removed() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        make_cfg(
            dataset={"dataset": "micro_puc"},
            model={"model": "graph_flare"},
        )


def test_geo_transolver_never_requires_edge_info() -> None:
    metadata = dict(c_in=3, c_out=1, space_dim=2)
    cfg = make_cfg(
        dataset={"dataset": "elasticity"},
        model={
            "model": "geo_transolver",
            "channel_dim": 32,
            "num_blocks": 1,
            "num_heads": 4,
            "num_slices": 4,
            "n_hidden_local": 8,
            "include_local_features": True,
            "ball_radii": (0.2,),
            "ball_ks": (4,),
        },
    )
    _, model = model_factory.make_model(cfg, metadata, GLOBAL_RANK=0)
    assert model.requires_edge_info is False


def test_old_graph_geo_transolver_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        make_cfg(model={"model": "graph_geo_transolver"})


def test_rigno_not_in_edge_info_models() -> None:
    assert "rigno" not in model_factory.EDGE_INFO_MODELS
    assert "graph_geo_transolver" not in model_factory.EDGE_INFO_MODELS
    assert model_factory.EDGE_INFO_MODELS == {"glt", "meshgraphnet", "gito"}
    metadata = dict(c_in=3, c_out=1, space_dim=2)
    cfg = make_cfg(
        dataset={"dataset": "elasticity"},
        model={"model": "rigno", "channel_dim": 32, "num_blocks": 1, "num_slices": 4},
    )
    _, model = model_factory.make_model(cfg, metadata, GLOBAL_RANK=0)
    assert model.requires_edge_info is False


def test_glt_default_raw_eigen_pe() -> None:
    from pdebench.models.graph_models.glt import GLT, GLTConfig
    from pdebench.models.graph_models.glt import RawEigenPE, RawEigenPEConfig

    model = GLT(
        GLTConfig(pe=RawEigenPEConfig(num_eigenmodes=8), pe_inject_mode="concat_qk"),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )
    assert isinstance(model.pe, RawEigenPE)
    assert model.pe.out_dim == 8


def test_glt_spe_pe_mode_parsing() -> None:
    from pdebench.models.graph_models.glt import GLT, GLTConfig
    from pdebench.models.graph_models.glt import SpectralFilterPE, SpectralFilterPEConfig, _parse_pe_spe_mode

    assert _parse_pe_spe_mode("query") == "query"
    assert _parse_pe_spe_mode("multihop") == "multihop"

    model = GLT(
        GLTConfig(pe=SpectralFilterPEConfig(num_eigenmodes=8, mode="multihop"), pe_inject_mode="concat_qk"),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )
    assert isinstance(model.pe, SpectralFilterPE)
    assert model.pe.spe_mode == "multihop"
    assert model.pe.norm_mode == "node_rms"


def test_glt_spe_builds_spectral_filter_pe() -> None:
    from torch import nn

    from pdebench.models.graph_models.glt import GLT, GLTConfig, SpectralFilterPE, SpectralFilterPEConfig

    model = GLT(
        GLTConfig(pe=SpectralFilterPEConfig(num_eigenmodes=8, filter_type="band"), pe_inject_mode="concat_qk"),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )
    assert isinstance(model.pe, SpectralFilterPE)
    assert model.pe.filter_type == "band"
    assert model.pe.spe_mode == "query"
    assert model.pe.out_dim == model.pe.out_feature_dim
    assert isinstance(model.c_proj, nn.Linear)


def test_glt_spe_spatial_filter_builds_spectral_filter_pe() -> None:
    from pdebench.models.graph_models.glt import GLT, GLTConfig
    from pdebench.models.graph_models.glt import SpectralFilterPE, SpectralFilterPEConfig

    model = GLT(
        GLTConfig(pe=SpectralFilterPEConfig(num_eigenmodes=8, filter_type="spatial"), pe_inject_mode="concat_qk"),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )
    assert isinstance(model.pe, SpectralFilterPE)
    assert model.pe.filter_type == "spatial"
    assert model.pe.poly_order == 2
