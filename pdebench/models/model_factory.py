from __future__ import annotations

import pdebench
from pdebench.dataset.ginot import GINOT_DATASETS
from pdebench.dataset.lpbf import LPBF_DATASETS
from pdebench.dataset.plaid_datasets import PLAID_DATASETS
from pdebench.dataset.registry import resolve_dataset_name
from pdebench.models.abupt_surface_mixer import ABUPTSurfaceMixerModel

#======================================================================#
ROLLOUT_DATASETS = {'navier_stokes', 'plasticity'}
SPLIT_INPUT_MODELS = {'transolver', 'lamo', 'gnot', 'mambano'}
BRANCH_INPUT_MODELS = {'lno'}
STATIC_MESH_DATASETS = PLAID_DATASETS | GINOT_DATASETS | LPBF_DATASETS
EDGE_INFO_MODELS = {
    'glt',
    'meshgraphnet',
    'gito',
}

CONV2D_STRUCTURED_MESH_DATASETS = frozenset({
    'navier_stokes',
    'plasticity',
    'darcy',
    'airfoil_steady',
    'pipe',
})


def _resolve_rmsnorm(rmsnorm, training_cfg) -> bool:
    if rmsnorm is None and training_cfg.mixed_precision:
        amp_name = str(training_cfg.amp_dtype or "bf16").lower()
        if amp_name in {"bf16", "bfloat16", "fp16", "float16", "half"}:
            rmsnorm = True
    return False if rmsnorm is None else bool(rmsnorm)


def _validate_conv2d_dataset(model_type: str, dataset: str, conv2d: bool) -> None:
    if not conv2d:
        return
    if dataset not in CONV2D_STRUCTURED_MESH_DATASETS:
        raise ValueError(
            f"model.conv2d=True for model={model_type!r} is only supported on structured 2D mesh "
            f"datasets {sorted(CONV2D_STRUCTURED_MESH_DATASETS)}, got {dataset!r}."
        )


def get_rollout_model_dims(cfg, metadata):
    point_input_dim = metadata['c_in']
    space_dim = metadata.get('space_dim', point_input_dim)
    fun_dim = metadata.get('fun_dim', 0)
    trunk_dim = space_dim
    branch_dim = fun_dim
    dataset = cfg.dataset.dataset
    model_type = cfg.model.model

    if dataset == 'plasticity':
        if model_type == 'mambano':
            fun_dim = fun_dim + 1
        elif model_type in BRANCH_INPUT_MODELS:
            trunk_dim = space_dim + 1
        elif model_type not in SPLIT_INPUT_MODELS:
            point_input_dim = point_input_dim + 1

    return dict(
        point_input_dim=point_input_dim,
        space_dim=space_dim,
        fun_dim=fun_dim,
        trunk_dim=trunk_dim,
        branch_dim=branch_dim,
    )


def wrap_rollout_model(cfg, model):
    dataset = cfg.dataset.dataset
    model_type = cfg.model.model
    if dataset == 'navier_stokes':
        return pdebench.NavierStokesModelAdapter(model, model_type)
    if dataset == 'plasticity':
        return pdebench.PlasticityModelAdapter(model, model_type)
    return model


def _apply_onecycle_recipe(
    training_cfg,
    optimizer_cfg,
    scheduler_cfg,
    *,
    learning_rate,
    opt_beta1,
    opt_beta2,
    pct_start,
    clip_grad_norm,
    weight_decay,
    optimizer=None,
    div_factor=25,
    final_div_factor=1e4,
    override_min_lr=None,
):
    if optimizer is not None:
        optimizer_cfg.optimizer = optimizer
    optimizer_cfg.learning_rate = learning_rate
    optimizer_cfg.opt_beta1 = opt_beta1
    optimizer_cfg.opt_beta2 = opt_beta2
    scheduler_cfg.schedule = 'OneCycleLR'
    scheduler_cfg.pct_start = pct_start
    scheduler_cfg.div_factor = div_factor
    scheduler_cfg.final_div_factor = final_div_factor
    scheduler_cfg.override_min_lr = override_min_lr
    training_cfg.clip_grad_norm = clip_grad_norm
    optimizer_cfg.weight_decay = weight_decay


#======================================================================#
def _resolve_model_spec(cfg, metadata):
    """Resolve constructor and args for cfg.model.

    Returns:
        tuple: (cfg, c_in, c_out, model_name, model_ctor)
    """

    dataset_cfg = cfg.dataset
    training_cfg = cfg.training
    optimizer_cfg = cfg.optimizer
    scheduler_cfg = cfg.scheduler
    model_cfg = cfg.model
    dataset = resolve_dataset_name(str(dataset_cfg.dataset).lower())
    model_type = model_cfg.model
    use_puri2025flare_config = cfg.use_puri2025flare_config

    c_in = metadata['c_in']
    c_out = metadata['c_out']
    dims = get_rollout_model_dims(cfg, metadata)
    point_input_dim = dims['point_input_dim']
    space_dim = dims['space_dim']
    fun_dim = dims['fun_dim']
    trunk_dim = dims['trunk_dim']
    branch_dim = dims['branch_dim']

    if model_type == 'transolver':
        #--------------------------------#
        # Transolver: https://arxiv.org/abs/2402.02366
        #--------------------------------#
        standard_benchmark_datasets = {'elasticity', 'navier_stokes', 'plasticity'}

        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )

            if dataset in standard_benchmark_datasets:
                training_cfg.mixed_precision = False
                training_cfg.compile_model = False
                training_cfg.ema = False
                training_cfg.grad_accumulation_steps = 1

            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 8
            elif (
                dataset in ['elasticity', 'shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            elif dataset in ['airfoil_steady', 'pipe', 'darcy']:
                training_cfg.batch_size = 4
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")

            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999

            if dataset == 'elasticity':
                scheduler_cfg.schedule = 'CosineAnnealingLR'
                training_cfg.clip_grad_norm = 0.1
            else:
                scheduler_cfg.schedule = 'OneCycleLR'
                scheduler_cfg.pct_start = 0.3
                scheduler_cfg.div_factor = 25
                scheduler_cfg.final_div_factor = 1e4
                scheduler_cfg.override_min_lr = None
                training_cfg.clip_grad_norm = None if dataset == 'navier_stokes' else 0.1

            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 1e-5

            # model params
            n_layers = 8
            n_hidden = 256 if dataset in ['airfrans', 'shapenet_car', 'navier_stokes'] else 128
            slice_num = 32 if dataset in ['airfrans', 'shapenet_car', 'navier_stokes'] else 64
            n_head = 8
            mlp_ratio = 1.0
            model_cfg.num_blocks = n_layers
            model_cfg.channel_dim = n_hidden
            model_cfg.num_heads = n_head
            model_cfg.num_slices = slice_num
            model_cfg.mlp_ratio = mlp_ratio

            if dataset == 'navier_stokes':
                model_cfg.unified_pos = True
            elif dataset == 'plasticity':
                model_cfg.unified_pos = False
            elif dataset == 'elasticity':
                model_cfg.unified_pos = False

        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            slice_num = model_cfg.num_slices
            n_head = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio

        _validate_conv2d_dataset(model_type, dataset, bool(model_cfg.conv2d))
        if model_cfg.conv2d:
            model_name = 'Transolver_Structured_Mesh_2D'
            Model = pdebench.Transolver_Structured_Mesh_2D
        else:
            model_name = 'Transolver'
            Model = pdebench.Transolver

    elif model_type in {
        'meshgraphnet',
        'rigno',
        'gito',
        'geo_transolver',
    }:
        #--------------------------------#
        # MeshGraphNet and graph operator variants
        #--------------------------------#
        if use_puri2025flare_config:
            model_cfg.act = 'silu'
            model_cfg.channel_dim = 128
            if model_type == 'gito':
                model_cfg.num_blocks_hgt = 2
                model_cfg.num_blocks_self_attn = 0
            else:
                model_cfg.num_blocks = 4

        if model_type == 'meshgraphnet':
            model_name = 'MeshGraphNetModel'
            Model = pdebench.MeshGraphNetModel
        elif model_type == 'rigno':
            model_name = 'RIGNOModel'
            Model = pdebench.RIGNOModel
        elif model_type == 'gito':
            model_name = 'GITOModel'
            Model = pdebench.GITOModel
        else:
            model_name = 'GeoTransolverModel'
            Model = pdebench.GeoTransolverModel

    elif model_type in ['set_transformer', 'set_transofmer']:
        #--------------------------------#
        # Set Transformer (ISAB backbone)
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif (
                dataset in ['elasticity', 'shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            elif dataset in ['airfoil_steady', 'pipe', 'darcy']:
                training_cfg.batch_size = 4
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")
            _apply_onecycle_recipe(
                training_cfg,
                optimizer_cfg,
                scheduler_cfg,
                learning_rate=1e-3,
                opt_beta1=0.9,
                opt_beta2=0.999,
                pct_start=0.3,
                clip_grad_norm=0.1,
                weight_decay=5e-2 if dataset == 'shapenet_car' else 1e-4 if dataset in ['drivaerml_40k', 'lpbf'] else 1e-5,
            )

            # model params (mirrors Transolver defaults)
            n_layers = 8
            n_hidden = 128 if dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 256
            slice_num = 64 if dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 32
            n_head = 8
            mlp_ratio = 2.0

        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            slice_num = model_cfg.num_slices
            n_head = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio

        model_name = 'SetTransformerModel'
        Model = pdebench.SetTransformerModel

    elif model_type == 'transolver++':
        #--------------------------------#
        # Transolver++
        #--------------------------------#
        if use_puri2025flare_config:
            # Table 10 of the Transolver++ paper publishes shared benchmark
            # defaults for epochs/optimizer/batch size and model width/slices.
            standard_benchmarks = {
                'elasticity': dict(batch_size=8, n_hidden=128, slice_num=64),
                'plasticity': dict(batch_size=8, n_hidden=128, slice_num=64),
                'airfoil_steady': dict(batch_size=4, n_hidden=128, slice_num=64),
                'pipe': dict(batch_size=4, n_hidden=128, slice_num=64),
                'navier_stokes': dict(batch_size=8, n_hidden=256, slice_num=32),
                'darcy': dict(batch_size=4, n_hidden=128, slice_num=64),
            }

            if dataset in standard_benchmarks:
                settings = standard_benchmarks[dataset]
                training_cfg.epochs = 500
                training_cfg.batch_size = settings['batch_size']
                n_layers = 8
                n_hidden = settings['n_hidden']
                slice_num = settings['slice_num']
                n_head = 8
            elif dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']:
                training_cfg.epochs = 250
                training_cfg.batch_size = 1
                n_layers = 8
                n_hidden = 128
                slice_num = 64
                n_head = 8
            else:
                raise ValueError(f"Transolver++ defaults not set for dataset {dataset}")

            _apply_onecycle_recipe(
                training_cfg,
                optimizer_cfg,
                scheduler_cfg,
                learning_rate=1e-3,
                opt_beta1=0.9,
                opt_beta2=0.999,
                pct_start=0.3,
                clip_grad_norm=0.1,
                optimizer='adamw',
                weight_decay=5e-2 if dataset == 'shapenet_car' else 1e-4 if dataset in ['drivaerml_40k', 'lpbf'] else 1e-5,
            )

            mlp_ratio = 1.0
        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            slice_num = model_cfg.num_slices
            n_head = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio

        model_name = 'TransolverPlusPlus'
        Model = pdebench.TransolverPlusPlus

    elif model_type == 'lno':
        #--------------------------------#
        # LNO: https://github.com/L-I-M-I-T/LatentNeuralOperator
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 4
            elif (
                dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")
            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.99
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.2
            scheduler_cfg.div_factor = 1e4
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 1000.0
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 5e-5

            # model params
            n_head = 8
            n_mode = 256
            n_dim = 192 if dataset in ['elasticity'] else 128
            n_layer = 3 if dataset in ['elasticity'] else 2
            n_block = 8 if dataset in ['pipe', 'airfoil_steady'] else 4

        else:
            n_head = model_cfg.num_heads
            n_mode = model_cfg.num_modes
            n_dim = model_cfg.channel_dim
            n_layer = model_cfg.num_layers_kv_proj
            n_block = model_cfg.num_blocks

        model_name = 'LNO'
        Model = pdebench.LNO

    elif model_type == 'gnot':
        #--------------------------------#
        # GNOT
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif dataset in ['elasticity']:
                training_cfg.batch_size = 2
            elif dataset in ['darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 4
            elif (
                dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")
            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.3
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 0.1
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in ['darcy']:
                optimizer_cfg.weight_decay = 5e-5
            else:
                optimizer_cfg.weight_decay = 1e-5

            # model params
            n_layers = 8
            n_hidden = 128
            mlp_ratio = 2.0
            n_experts = 3
            n_head = 8
        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            mlp_ratio = model_cfg.mlp_ratio
            n_experts = model_cfg.num_experts
            n_head = model_cfg.num_heads

        if dataset in ['darcy', 'airfoil_steady', 'pipe', 'navier_stokes', 'plasticity']:
            geotype = 'structured_2D'
            unified_pos = True
            ref = 8
            shapelist = [metadata['H'], metadata['W']]
        elif (
            dataset in ['elasticity', 'shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
            or dataset.startswith('drivaerml')
            or dataset in GINOT_DATASETS
        ):
            geotype = 'unstructured'
            unified_pos = False
            ref = 8
            shapelist = None
        else:
            raise ValueError(f"Geotype not set for dataset {dataset}")

        model_name = 'GNOT'
        Model = pdebench.GNOT

    elif model_type == 'abupt_surface_mixer':
        #--------------------------------#
        # Isolated AB-UPT surface mixer
        #--------------------------------#
        model_name = 'AB-UPT-Surface-Mixer'
        Model = ABUPTSurfaceMixerModel

    elif model_type == 'upt':
        #--------------------------------#
        # UPT (Universal Physics Transformer)
        #--------------------------------#
        raise NotImplementedError("UPT is not implemented yet.")

    elif model_type == 'lamo':
        #--------------------------------#
        # LaMO
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
                model_cfg.unified_pos = True
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
                model_cfg.unified_pos = False
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 4
            elif (
                dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")

            n_layers = 8
            n_hidden = 128 if dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 256
            slice_num = 64 if dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 32
            n_head = 8
            mlp_ratio = 1.0

            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.3
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 0.1
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 1e-5
        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            slice_num = model_cfg.num_slices
            n_head = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio

        _validate_conv2d_dataset(model_type, dataset, bool(model_cfg.conv2d))
        if model_cfg.conv2d:
            model_name = 'LaMO_Structured_Mesh_2D'
            Model = pdebench.LaMO_Structured_Mesh_2D
        else:
            model_name = 'LaMO'
            Model = pdebench.LaMO

    elif model_type == 'mambano':
        #--------------------------------#
        # MambaNO
        #--------------------------------#
        if model_cfg.compile_model:
            print(
                "Disabling torch.compile for mambano due torch._dynamo recompilation instability "
                "with selective_scan custom CUDA ops."
            )
            training_cfg.compile_model = False
            training_cfg.static_graph = False

        allowed_structured_datasets = {'darcy', 'pipe', 'airfoil_steady', 'navier_stokes', 'plasticity'}
        if dataset not in allowed_structured_datasets:
            raise ValueError(
                f"MambaNO currently supports only structured 2D datasets "
                f"{sorted(allowed_structured_datasets)}, got '{dataset}'."
            )

        if use_puri2025flare_config:
            training_cfg.epochs = 250 if dataset in ['shapenet_car', 'lpbf'] else 500
            n_layers = 8
            n_hidden = 16
            d_state = 16
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 4
            elif dataset == 'plasticity':
                training_cfg.batch_size = 8
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 4
            elif dataset in ['shapenet_car', 'lpbf'] or dataset.startswith('drivaerml') or dataset in GINOT_DATASETS:
                training_cfg.batch_size = 1

            optimizer_cfg.learning_rate = 5e-4
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.3
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 0.1
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 1e-5
        else:
            n_layers = model_cfg.num_blocks
            n_hidden = model_cfg.channel_dim
            d_state = model_cfg.mambano_d_state

        stage_depth = max(1, n_layers // 4)
        depths = [stage_depth] * 4

        model_name = 'MambaNO_Structured_Mesh_2D'
        Model = pdebench.MambaNO_Structured_Mesh_2D

    elif model_type == 'gaot':
        #--------------------------------#
        # GAOT (PDEBench-adapted)
        #--------------------------------#
        if use_puri2025flare_config:
            # Upstream elasticity defaults mirrored from run_gaot.sh:
            # epoch=1000, batch_size=64, lr=8e-4, wd=1e-5,
            # hidden/lifting=64, layers=3, heads=8, patch=2, latent=64x64.
            training_cfg.epochs = 500

            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 4
            elif dataset in ['shapenet_car', 'lpbf'] or dataset.startswith('drivaerml') or dataset in GINOT_DATASETS:
                training_cfg.batch_size = 1

            optimizer_cfg.learning_rate = 8e-4
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.3
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 0.1
            optimizer_cfg.weight_decay = 1e-5
            # GAOT neighbor search includes Python-side control flow that causes
            # frequent graph breaks under torch.compile.
            training_cfg.compile_model = False
            training_cfg.static_graph = False

            magno_cfg = model_cfg.args.magno
            tr_cfg = model_cfg.args.transformer
            magno_cfg.coord_dim = 2
            magno_cfg.lifting_channels = 32
            magno_cfg.use_geoembed = True
            magno_cfg.neighbor_search_method = 'auto'
            magno_cfg.use_torch_scatter = True
            tr_cfg.num_layers = 3
            tr_cfg.patch_size = 2
            tr_cfg.num_heads = 8
            tr_cfg.ffn_multiplier = 4.0
            model_cfg.latent_tokens_size = [64, 64]

        model_name = 'GAOT_PDEBench'
        Model = pdebench.GAOT_PDEBench

    elif model_type == 'perceiverio':
        #--------------------------------#
        # PerceiverIO
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 2
            elif (
                dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")
            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.1
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 1.0
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 1e-5

            # model params
            channel_dim = 128
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None
            num_latents = 512
            cross_attn = model_cfg.pcvr_cross_attn
        else:
            channel_dim = model_cfg.channel_dim
            num_blocks = model_cfg.num_blocks
            num_heads = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio
            act = model_cfg.act
            num_latents = model_cfg.num_latents
            cross_attn = model_cfg.pcvr_cross_attn

        model_name = 'PerceiverIO'
        Model = pdebench.PerceiverIO

    elif model_type in {'transformer', 'glt'}:
        #--------------------------------#
        # Vanilla Transformer / graph topology transformers
        #--------------------------------#
        if use_puri2025flare_config:
            training_cfg.epochs = (
                250 if dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf'] else 500
            )
            if dataset == 'navier_stokes':
                training_cfg.batch_size = 2
            elif dataset == 'plasticity':
                training_cfg.batch_size = 4
            elif dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                training_cfg.batch_size = 2
            elif (
                dataset in ['shapenet_car', 'nasa_crm', 'ahmedml_surface', 'drivaerml_surface', 'lpbf']
                or dataset.startswith('drivaerml')
                or dataset in GINOT_DATASETS
            ):
                training_cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {dataset}")
            
            # training params
            optimizer_cfg.optimizer = 'adamw'
            optimizer_cfg.learning_rate = 1e-3
            optimizer_cfg.opt_beta1 = 0.9
            optimizer_cfg.opt_beta2 = 0.999
            optimizer_cfg.opt_eps = 1e-6 if dataset in ['pipe'] else 1e-8
            scheduler_cfg.schedule = 'OneCycleLR'
            scheduler_cfg.pct_start = 0.1
            scheduler_cfg.div_factor = 25
            scheduler_cfg.final_div_factor = 1e4
            scheduler_cfg.override_min_lr = None
            training_cfg.clip_grad_norm = 1.0
            if dataset in ['shapenet_car']:
                optimizer_cfg.weight_decay = 5e-2
            elif dataset in ['drivaerml_40k']:
                optimizer_cfg.weight_decay = 1e-4
            elif dataset in LPBF_DATASETS:
                optimizer_cfg.weight_decay = 1e-4
            else:
                optimizer_cfg.weight_decay = 1e-5

            # model params
            channel_dim = 80
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None
            rmsnorm = False

        else:
            channel_dim = model_cfg.channel_dim
            num_blocks = model_cfg.num_blocks
            num_heads = model_cfg.num_heads
            mlp_ratio = model_cfg.mlp_ratio
            act = model_cfg.act
            rmsnorm = model_cfg.rmsnorm

        if model_type == 'glt':
            model_cfg.rmsnorm = _resolve_rmsnorm(rmsnorm, training_cfg)
            if dataset not in STATIC_MESH_DATASETS:
                raise ValueError(f"model_type='{model_type}' is currently only supported for static mesh datasets.")
            model_name = 'GLT'
            Model = pdebench.GLT
        else:
            backend_kwargs = dict(
                mlp_ratio=mlp_ratio,
            )

            model_name = 'Transformer'
            Model = pdebench.TransformerWrapper

    elif model_type == 'linformer':
        #--------------------------------#
        # Linformer
        #--------------------------------#
        backend_kwargs = dict(
            mlp_ratio=model_cfg.mlp_ratio,
            seq_len=metadata['max_length'],
            k=model_cfg.linformer_k,
        )
        
        model_name = 'Linformer'
        Model = pdebench.TransformerWrapper

    elif model_type == 'linear':
        #--------------------------------#
        # Linear attention
        #--------------------------------#
        backend_kwargs = dict(
            mlp_ratio=model_cfg.mlp_ratio,
            kernel=model_cfg.kernel,
            norm_q=model_cfg.norm_q,
            norm_k=model_cfg.norm_k,
        )

        model_name = 'Linear'
        Model = pdebench.TransformerWrapper

    elif model_type == 'flare':
        #--------------------------------#
        # FLARE
        #--------------------------------#
        channel_dim = model_cfg.channel_dim
        num_heads = model_cfg.num_heads
        num_latents = model_cfg.num_latents
        attn_scale = model_cfg.attn_scale
        if isinstance(attn_scale, str):
            assert attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {attn_scale}. Choose from: sqrt, one."
        else:
            attn_scale = float(attn_scale)
        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."
        head_dim = channel_dim // num_heads
       
        if head_dim > 16:
            attn_scale = 'sqrt'

        if isinstance(attn_scale, str):
            attn_scale = (head_dim ** -0.5) if attn_scale == 'sqrt' else 1.0
        model_cfg.attn_scale = attn_scale

        backend_kwargs = dict(
            attn_scale=attn_scale,
            num_latents=num_latents,
            num_layers_k_proj=model_cfg.num_layers_k_proj,
            num_layers_v_proj=model_cfg.num_layers_v_proj,
            k_proj_mlp_ratio=model_cfg.k_proj_mlp_ratio,
            v_proj_mlp_ratio=model_cfg.v_proj_mlp_ratio,
            qk_norm=model_cfg.qk_norm,
        )

        model_name = 'FLARE'
        Model = pdebench.FLAREModel

    elif model_type == 'flare_experimental':
        #--------------------------------#
        # FLARE
        #--------------------------------#
        assert model_cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {model_cfg.attn_scale}. Choose from: sqrt, one."
        assert model_cfg.channel_dim % model_cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {model_cfg.channel_dim} and {model_cfg.num_heads}."
        head_dim = model_cfg.channel_dim // model_cfg.num_heads
        model_cfg.attn_scale = (head_dim ** -0.5) if model_cfg.attn_scale == 'sqrt' else 1.0

        backend_kwargs = dict(
            attn_scale=model_cfg.attn_scale,
            num_latents=model_cfg.num_latents,
            num_layers_k_proj=model_cfg.num_layers_k_proj,
            num_layers_v_proj=model_cfg.num_layers_v_proj,
            k_proj_mlp_ratio=model_cfg.k_proj_mlp_ratio,
            v_proj_mlp_ratio=model_cfg.v_proj_mlp_ratio,
            num_layers_ffn=model_cfg.num_layers_ffn,
            ffn_mlp_ratio=model_cfg.ffn_mlp_ratio,
            qk_norm=model_cfg.qk_norm,
        )

        model_name = 'FLARE-Experimental'
        Model = pdebench.FLAREExperimentalModel

    elif model_type == 'flare_ablations':
        #--------------------------------#
        # BigFLARE (ablations)
        #--------------------------------#
        model_name = 'BigFLARE'
        Model = pdebench.BigFLAREModel

    elif model_type == 'mixer_backbone':
        #--------------------------------#
        # Mixer backbone (identical FLARE backbone)
        #--------------------------------#
        # Training hparams: FLARE defaults (no Transolver use_puri2025flare_config recipe).
        # FLARE/FLAREPP attn_scale is hardcoded to head_dim ** -0.5 inside the mixers.
        model_cfg.rmsnorm = _resolve_rmsnorm(model_cfg.rmsnorm, training_cfg)

        model_name = 'MixerBackbone'
        Model = pdebench.MixerBackboneModel

    elif model_type == 'flarepp':
        #--------------------------------#
        # FLAREPP
        #--------------------------------#
        # Training hparams: same fall-through as FLARE (no special recipe).
        model_name = 'FLAREPP'
        Model = pdebench.FLAREPPModel

    elif model_type == 'luna':
        #--------------------------------#
        # Luna encoder (pack / unpack)
        #--------------------------------#
        model_name = 'Luna'
        Model = pdebench.LunaModel

    else:
        #--------------------------------#
        # No model selected
        #--------------------------------#
        raise NotImplementedError(f"Model type {model_type} not implemented.")

    return cfg, c_in, c_out, model_name, Model


def make_model(cfg, metadata, GLOBAL_RANK):
    metadata = dict(metadata)
    metadata.setdefault("dataset", cfg.dataset.dataset)
    metadata.setdefault("model", cfg.model.model)
    cfg, c_in, c_out, model_name, Model = _resolve_model_spec(cfg, metadata)
    training_cfg = cfg.training
    model_cfg = cfg.model

    if GLOBAL_RANK == 0:
        print(f"Using {model_name}(c_in={c_in}, c_out={c_out}) with config={model_cfg}")

    if cfg.use_puri2025flare_config and cfg.dataset.dataset in {'navier_stokes', 'plasticity'}:
        training_cfg.batch_size = 2

    if cfg.model.model == "glt":
        metadata.setdefault("pos_dim", int(metadata.get("space_dim", metadata.get("c_in", 1))))

    model = Model(model_cfg, metadata=metadata)
    model = wrap_rollout_model(cfg, model)
    model.requires_edge_info = cfg.model.model in EDGE_INFO_MODELS

    return cfg, model
