from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from types import UnionType
from typing import List, Optional, Union, get_args, get_origin, get_type_hints

from pdebench.dataset.ginot import GINOT_DATASETS
from pdebench.models.abupt_surface_mixer import ABUPTSurfaceMixerConfig
from pdebench.models.flare import FlareConfig
from pdebench.models.flare_ablations import FlareAblationsConfig
from pdebench.models.flare_experimental import FlareExperimentalConfig
from pdebench.models.flarepp import FlarePPConfig
from pdebench.models.gaot import GAOTConfig
from pdebench.models.gnot import GNOTConfig
from pdebench.models.graph_models.geo_transolver import GeoTransolverConfig
from pdebench.models.graph_models.gito import GITOConfig
from pdebench.models.graph_models.glt import GRAPH_PE_BY_KIND, GLTConfig
from pdebench.models.graph_models.meshgraphnets import MeshGraphNetConfig
from pdebench.models.graph_models.rigno import RIGNOConfig
from pdebench.models.lamo import LaMOConfig
from pdebench.models.lno import LNOConfig
from pdebench.models.luna import LunaConfig
from pdebench.models.mambano import MambaNOConfig
from pdebench.models.mixer_backbone import MIXER_BY_KIND, MixerBackboneConfig
from pdebench.models.perceiver import PerceiverIOConfig
from pdebench.models.set_transformer import SetTransformerConfig
from pdebench.models.transformer import LinearConfig, LinformerConfig, TransformerConfig
from pdebench.models.transolver import TransolverConfig
from pdebench.models.transolver_plus import TransolverPlusPlusConfig
from pdebench.models.upt import UPTConfig


@dataclass
class RunConfig:
    train: bool = False
    evaluate: bool = False
    restart: bool = False
    load_weights_path: Optional[str] = None
    exp_name: str = "exp"
    seed: int = 0
    deterministic: bool = False
    timing_only: bool = False

@dataclass
class DatasetConfig:
    dataset: Optional[str] = None
    data_root: Optional[str] = None
    # After the train/test split, keep only the first N indices per split (0 = full split).
    max_samples: int = 0
    # Static PLAID hyperelasticity / el-pl dynamics: deploy SDF/projection-vector features at assemble time.
    # elpl_v3 always stores SDF in cache; this flag gates runtime graph assembly only (no cache rebuild).
    plaid_use_sdf_features: bool = True
    # plaid_elpl_terminal only: runtime y target normalization (default asinh_iqr; cache = z-score).
    plaid_terminal_y_norm: str = "asinh_iqr"
    # plaid_elpl_terminal only: predict subset of benchmark fields (default U_x only).
    plaid_terminal_target_fields: List[str] = field(default_factory=lambda: ["U_x"])
    # Static PLAID only: build the unlabeled public test split for HF submission.
    plaid_load_public_test: bool = False
    # ahmedml_surface / drivaerml_surface: cells per amortized train/test step
    # (fullbatch uses full mesh). Default IID for both; False → strided parts.
    subset_size: int = 100_000
    # ahmedml_surface / drivaerml_surface: True → IID train subsets (sorted); False → strided.
    iid_samples: bool = True
    # ahmedml_surface / drivaerml_surface: False → normalized MSE; True →
    # 0.5 * Rel-L2(pressure) + 0.5 * Rel-L2(wall-shear magnitude) in physical units.
    # Applies to both per-batch train loss and full-mesh train/test loss.
    rel_l2_loss: bool = False


@dataclass
class GinotDatasetConfig(DatasetConfig):
    mesh_split_seed: int = 0


@dataclass
class TrainingConfig:
    epochs: int = 100
    steps: int = 0
    stats_every: int = 0
    stats_on_start: bool = True
    fullbatch_stats_on_start: bool = False
    fullbatch_stats_train: bool = True
    fullbatch_stats_test: bool = True
    batch_size: int = 1  # global batch (total graphs per optimizer step across DDP ranks)
    clip_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    mixed_precision: bool = False
    amp_dtype: Optional[str] = None
    compile_model: bool = True
    static_graph: bool = True
    ema: bool = True
    ema_decay: float = 0.999
    num_workers: int = 8
    prefetch_factor: Optional[int] = None
    overlap_train_dataloader: bool = True
    continuous_train_batches: Optional[bool] = None
    use_context_parallel: bool = False
    context_parallel_size: int = 1
    cp_sequence_dim: int = 1
    cp_debug_gather_outputs: bool = False


@dataclass
class OptimizerConfig:
    optimizer: str = "adamw"
    learning_rate: Union[float, List[float]] = 1e-3
    weight_decay: Union[float, List[float]] = 0.0
    # Absolute AdamW decay for GLT c_proj + blocks_c (dual stream). None = same as weight_decay.
    glt_c_stream_weight_decay: Optional[float] = None
    opt_beta1: Union[float, List[float]] = 0.9
    opt_beta2: Union[float, List[float]] = 0.999
    opt_eps: Union[float, List[float]] = 1e-8

@dataclass
class SchedulerConfig:
    schedule: Optional[str] = "OneCycleLR"


@dataclass
class OneCycleSchedulerConfig(SchedulerConfig):
    pct_start: float = 0.10
    div_factor: float = 1e4
    final_div_factor: float = 1e4
    three_phase: bool = False
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    anneal_strategy: str = "cos"
    override_min_lr: Optional[float] = None


@dataclass
class PlateauSchedulerConfig(SchedulerConfig):
    plateau_factor: float = 0.7
    plateau_patience: int = 10


@dataclass
class CosineAnnealingSchedulerConfig(SchedulerConfig):
    schedule: Optional[str] = "CosineAnnealingLR"
    min_lr: float = 0.0


@dataclass
class ConstantSchedulerConfig(SchedulerConfig):
    schedule: Optional[str] = None


@dataclass
class ModelConfig:
    model: str = "flare"

MODEL_CONFIG_BY_MODEL = {
    "abupt_surface_mixer": ABUPTSurfaceMixerConfig,
    "transolver": TransolverConfig,
    "meshgraphnet": MeshGraphNetConfig,
    "rigno": RIGNOConfig,
    "gito": GITOConfig,
    "geo_transolver": GeoTransolverConfig,
    "set_transformer": SetTransformerConfig,
    "set_transofmer": SetTransformerConfig,
    "transolver++": TransolverPlusPlusConfig,
    "lno": LNOConfig,
    "gnot": GNOTConfig,
    "upt": UPTConfig,
    "lamo": LaMOConfig,
    "mambano": MambaNOConfig,
    "gaot": GAOTConfig,
    "perceiverio": PerceiverIOConfig,
    "transformer": TransformerConfig,
    "glt": GLTConfig,
    "linformer": LinformerConfig,
    "linear": LinearConfig,
    "flare": FlareConfig,
    "flare_experimental": FlareExperimentalConfig,
    "flare_ablations": FlareAblationsConfig,
    "mixer_backbone": MixerBackboneConfig,
    "flarepp": FlarePPConfig,
    "luna": LunaConfig,
}


SCHEDULER_CONFIG_BY_SCHEDULE = {
    None: ConstantSchedulerConfig,
    "ConstantLR": ConstantSchedulerConfig,
    "OneCycleLR": OneCycleSchedulerConfig,
    "ReduceLROnPlateau": PlateauSchedulerConfig,
    "CosineAnnealingLR": CosineAnnealingSchedulerConfig,
}


TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
FALSE_STRINGS = {"false", "0", "no", "n", "off"}
SCALAR_TYPES = {int, float, str, bool}


def _coerce_scalar(value, target_type):
    if value is None:
        return None

    if target_type is bool and isinstance(value, str):
        lowered = value.lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False

    if target_type in SCALAR_TYPES and isinstance(value, str):
        return target_type(value)

    return value


def _non_none_union_args(annotation):
    return [arg for arg in get_args(annotation) if arg is not type(None)]


def _is_list_annotation(annotation):
    return get_origin(annotation) in {list, List}


def _coerce_value(value, annotation, default=MISSING):
    if value is None:
        return None

    origin = get_origin(annotation)

    if origin in {Union, UnionType}:
        non_none_args = _non_none_union_args(annotation)
        if isinstance(value, list):
            list_args = [arg for arg in non_none_args if _is_list_annotation(arg)]
            if list_args:
                return _coerce_value(value, list_args[0], default)

        for arg in non_none_args:
            if _is_list_annotation(arg):
                continue
            coerced = _coerce_value(value, arg, default)
            if coerced is not value or isinstance(coerced, arg):
                return coerced

        return value

    if _is_list_annotation(annotation):
        args = get_args(annotation)
        item_type = args[0] if args else str
        if isinstance(value, list):
            return [_coerce_value(item, item_type) for item in value]
        return [_coerce_value(value, item_type)]

    if annotation in SCALAR_TYPES:
        return _coerce_scalar(value, annotation)

    if default is not MISSING and default is not None:
        return _coerce_scalar(value, type(default))

    return value


def _field_names(cls):
    return {f.name for f in fields(cls)}


def _check_dataclass_keys(data, cls):
    field_names = _field_names(cls)
    unknown = sorted(set(data) - field_names)
    if unknown:
        unknown_text = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Unknown field(s) for {cls.__name__}: {unknown_text}")


def _filtered_dataclass_data(data, cls):
    field_names = _field_names(cls)
    return {key: value for key, value in data.items() if key in field_names}


def _typed_dataclass_kwargs(data, cls):
    type_hints = get_type_hints(cls)
    out = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        out[f.name] = _coerce_value(data[f.name], type_hints.get(f.name, f.type), f.default)
    return out


def _build_dataclass(cls, data):
    _check_dataclass_keys(data, cls)
    return cls(**_typed_dataclass_kwargs(_filtered_dataclass_data(data, cls), cls))


_GRAPH_PE_CONFIG_CLASSES = tuple(config_cls for config_cls, _, _ in GRAPH_PE_BY_KIND.values())
_MIXER_CONFIG_CLASSES = tuple(config_cls for config_cls, _ in MIXER_BY_KIND.values())


def _unflatten_prefixed_keys(data: dict, prefix: str) -> dict:
    """Nest ``f"{prefix}.field"`` CLI keys (e.g. ``--model.pe.kind``) under ``prefix``.

    jsonargparse flattens dict-typed fields to dotted string keys (``"pe.kind"``) rather
    than nesting them, since ``Config.model`` is only annotated as ``ModelConfig | dict``
    at parse time. Merge those flattened keys into any existing dict already at ``prefix``
    (e.g. from a single ``--model.pe='{"kind": ...}'`` JSON literal).
    """
    flat_prefix = f"{prefix}."
    nested = dict(data[prefix]) if isinstance(data.get(prefix), dict) else {}
    out = {}
    for key, value in data.items():
        if key == prefix:
            continue
        if key.startswith(flat_prefix):
            nested[key[len(flat_prefix):]] = value
        else:
            out[key] = value
    if nested:
        out[prefix] = nested
    return out


def _coerce_glt_pe_config(value):
    if isinstance(value, _GRAPH_PE_CONFIG_CLASSES):
        return value
    if not isinstance(value, dict):
        return value
    pe_data = dict(value)
    kind = pe_data.get("kind", "raw_eigen")
    if kind not in GRAPH_PE_BY_KIND:
        raise ValueError(f"Unknown GLT pe kind={kind!r}. Choose from: {sorted(GRAPH_PE_BY_KIND)}")
    pe_cls, _, _ = GRAPH_PE_BY_KIND[kind]
    return _build_dataclass(pe_cls, pe_data)


def _coerce_mixer_config(value):
    if isinstance(value, _MIXER_CONFIG_CLASSES):
        return value
    if not isinstance(value, dict):
        return value
    mixer_data = dict(value)
    kind = mixer_data.get("kind", "flare")
    if kind not in MIXER_BY_KIND:
        raise ValueError(f"Unknown mixer kind={kind!r}. Choose from: {sorted(MIXER_BY_KIND)}")
    mixer_cls, _ = MIXER_BY_KIND[kind]
    return _build_dataclass(mixer_cls, mixer_data)


def _specialize_dataclass_instance(value, cls):
    defaults = cls()
    kwargs = {
        f.name: getattr(value, f.name, getattr(defaults, f.name))
        for f in fields(cls)
    }
    if cls is GLTConfig:
        kwargs["pe"] = _coerce_glt_pe_config(kwargs["pe"])
    if cls is MixerBackboneConfig:
        kwargs["mixer"] = _coerce_mixer_config(kwargs["mixer"])
    return cls(**kwargs)


def _coerce_dataclass_config(value, cls):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        return value
    return _build_dataclass(cls, value)


def _coerce_dataset_config(value):
    if isinstance(value, DatasetConfig) and type(value) is not DatasetConfig:
        return value
    if type(value) is DatasetConfig:
        dataset_name = value.dataset
        if dataset_name in GINOT_DATASETS:
            return _specialize_dataclass_instance(value, GinotDatasetConfig)
        return value

    if not isinstance(value, dict):
        return value

    dataset_data = dict(value)
    dataset_name = dataset_data.get("dataset")
    dataset_cls = GinotDatasetConfig if dataset_name in GINOT_DATASETS else DatasetConfig
    return _build_dataclass(dataset_cls, dataset_data)


def _coerce_scheduler_config(value):
    if isinstance(value, SchedulerConfig) and type(value) is not SchedulerConfig:
        return value

    if type(value) is SchedulerConfig:
        schedule = value.schedule
        scheduler_cls = SCHEDULER_CONFIG_BY_SCHEDULE.get(schedule, OneCycleSchedulerConfig)
        return _specialize_dataclass_instance(value, scheduler_cls)

    if not isinstance(value, dict):
        return value

    scheduler_data = dict(value)
    schedule = scheduler_data.get("schedule", "OneCycleLR")
    scheduler_cls = SCHEDULER_CONFIG_BY_SCHEDULE.get(schedule, OneCycleSchedulerConfig)
    return _build_dataclass(scheduler_cls, _filtered_dataclass_data(scheduler_data, scheduler_cls))


def _coerce_model_config(value):
    if isinstance(value, ModelConfig) and type(value) is not ModelConfig:
        if isinstance(value, GLTConfig):
            value.pe = _coerce_glt_pe_config(value.pe)
        if isinstance(value, MixerBackboneConfig):
            value.mixer = _coerce_mixer_config(value.mixer)
        return value

    if type(value) is ModelConfig:
        model_type = value.model
        model_cls = _model_config_cls(model_type)
        return _specialize_dataclass_instance(value, model_cls)

    if not isinstance(value, dict):
        return value

    model_data = dict(value)
    model_type = model_data.get("model", "flare")
    model_cls = _model_config_cls(model_type)
    if model_cls is GLTConfig:
        model_data = _unflatten_prefixed_keys(model_data, "pe")
    if model_cls is MixerBackboneConfig:
        model_data = _unflatten_prefixed_keys(model_data, "mixer")
    built = _build_dataclass(model_cls, model_data)
    if model_cls is GLTConfig:
        built.pe = _coerce_glt_pe_config(built.pe)
    if model_cls is MixerBackboneConfig:
        built.mixer = _coerce_mixer_config(built.mixer)
    return built


def _model_config_cls(model_type):
    if model_type not in MODEL_CONFIG_BY_MODEL:
        raise ValueError(f"Unknown model={model_type!r}. Choose from: {sorted(MODEL_CONFIG_BY_MODEL)}")
    return MODEL_CONFIG_BY_MODEL[model_type]


@dataclass
class Config:
    run: RunConfig | dict = field(default_factory=RunConfig)
    dataset: DatasetConfig | dict = field(default_factory=DatasetConfig)
    training: TrainingConfig | dict = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig | dict = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | dict = field(default_factory=SchedulerConfig)
    model: ModelConfig | dict = field(default_factory=ModelConfig)
    use_puri2025flare_config: bool = False

    def __post_init__(self):
        self.run = _coerce_dataclass_config(self.run, RunConfig)
        self.dataset = _coerce_dataset_config(self.dataset)
        self.training = _coerce_dataclass_config(self.training, TrainingConfig)
        self.optimizer = _coerce_dataclass_config(self.optimizer, OptimizerConfig)
        self.scheduler = _coerce_scheduler_config(self.scheduler)
        self.model = _coerce_model_config(self.model)

    def to_dict(self) -> dict:
        return asdict(self)
