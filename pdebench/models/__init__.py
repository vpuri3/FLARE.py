#
import warnings

from .abupt_surface_mixer import ABUPTSurfaceMixerConfig, ABUPTSurfaceMixerModel
from .flare import *
from .flarepp import *
from .flare_ablations import *
from .mixer_backbone import *
from .flare_experimental import *
from .gaot import *
from .gnot import *
from .graph_models.gito import *
from .graph_models.geo_transolver import *
from .graph_models.meshgraphnets import *
from .graph_models.rigno import *
from .graph_models.glt import *
from .lno import *
from .loopy import *
from .luna import *
from .perceiver import *
from .set_transformer import *
from .transformer import *
from .transolver import *
from .transolver_plus import *
from .unloopy import *
from .upt import *

try:
    from .lamo import *
except ModuleNotFoundError as exc:
    _LAMO_IMPORT_ERROR = exc
    _LAMO_WARNED = False

    def _warn_missing_lamo_once():
        global _LAMO_WARNED
        if not _LAMO_WARNED:
            warnings.warn(
                f"Optional LaMO dependencies are unavailable ({_LAMO_IMPORT_ERROR}). "
                "Install mamba-ssm/causal-conv1d to enable LaMO models.",
                RuntimeWarning,
                stacklevel=3,
            )
            _LAMO_WARNED = True

    class LaMO:  # type: ignore[no-redef]
        def __init__(self, config: "LaMOConfig", metadata=None):
            _warn_missing_lamo_once()
            raise ModuleNotFoundError(
                "LaMO is unavailable because optional dependencies are missing "
                "(mamba-ssm/causal-conv1d)."
            ) from _LAMO_IMPORT_ERROR

    class LaMO_Structured_Mesh_2D:  # type: ignore[no-redef]
        def __init__(self, config: "LaMOConfig", metadata=None):
            _warn_missing_lamo_once()
            raise ModuleNotFoundError(
                "LaMO_Structured_Mesh_2D is unavailable because optional dependencies are missing "
                "(mamba-ssm/causal-conv1d)."
            ) from _LAMO_IMPORT_ERROR

try:
    from .mambano import *
except ModuleNotFoundError as exc:
    _MAMBANO_IMPORT_ERROR = exc
    _MAMBANO_WARNED = False

    def _warn_missing_mambano_once():
        global _MAMBANO_WARNED
        if not _MAMBANO_WARNED:
            warnings.warn(
                f"Optional MambaNO dependencies are unavailable ({_MAMBANO_IMPORT_ERROR}). "
                "Install mamba-ssm/causal-conv1d to enable MambaNO models.",
                RuntimeWarning,
                stacklevel=3,
            )
            _MAMBANO_WARNED = True

    class MambaNO_Structured_Mesh_2D:  # type: ignore[no-redef]
        def __init__(self, config: "MambaNOConfig", metadata=None):
            _warn_missing_mambano_once()
            raise ModuleNotFoundError(
                "MambaNO_Structured_Mesh_2D is unavailable because optional dependencies are missing "
                "(mamba-ssm/causal-conv1d)."
            ) from _MAMBANO_IMPORT_ERROR
