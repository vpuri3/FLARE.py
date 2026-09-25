#
import os
import socket
import sys
import time

import torch
import yaml
from jsonargparse import ArgumentParser

import mlutils

# local
import pdebench
from pdebench.callbacks import (
    make_ahmedml_surface_statsfun,
    make_drivaerml_surface_statsfun,
    make_navier_stokes_statsfun,
    make_plasticity_statsfun,
    surface_batch_loss,
)
from pdebench.config import Config
from pdebench.dataset.ginot import (
    ginot_model_forward,
    ginot_postprocess_displacement,
    make_ginot_statsfun,
)
from pdebench.dataset.loss import compute_packed_loss
from pdebench.dataset.lpbf import (
    LPBF_DATASETS,
    lpbf_flare_batch_loss,
    lpbf_warped_rel_l2,
    make_lpbf_collate_fn,
    make_lpbf_statsfun,
    resolve_lpbf_batch_format,
)
from pdebench.dataset.mesh_runtime import (
    MESH_GRAPH_MODELS,
    MESH_SEQUENCE_MODELS,
    make_mesh_static_statsfun,
    mesh_batch_plaid_scaled_mse,
    mesh_model_forward,
    mesh_sequence_collate_fn,
    mesh_static_supports_time_cond,
)
from pdebench.dataset.plaid_datasets import PLAID_DATASETS
from pdebench.dataset.registry import resolve_dataset_name
from pdebench.dataset.sample import FeatureRequest, LossSpec
from pdebench.dataset.utils import (
    compile_stats_model_for_dataset,
    resolve_use_flash_varlen,
    uses_ginot_pipeline,
)
from pdebench.distributed import (
    build_context_parallel_state,
    cp_reduced_mse_loss,
    cp_reduced_rel_l2_loss,
    shard_batch,
)
from pdebench.models.model_factory import EDGE_INFO_MODELS, make_model

STATIC_MESH_DATASETS = PLAID_DATASETS
MSE_NORMALIZED_DATASETS = frozenset({"nasa_crm", "ahmedml_surface", "drivaerml_surface"})

#======================================================================#
PROJDIR = mlutils.dotdot(os.path.dirname(__file__))
OUTNAME = os.path.basename(os.path.dirname(__file__))
CASEDIR = os.path.join(PROJDIR, 'out', OUTNAME)

mlutils.set_cache_path(mlutils.dotdot(PROJDIR))
os.makedirs(CASEDIR, exist_ok=True)

MACHINE = socket.gethostname()
if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti 11 GB
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')


def main(cfg, device, *, run_timer: mlutils.RunTimer | None = None):
    timer = run_timer or mlutils.disabled_timer()
    timer.mark("main_enter")

    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    run_cfg = cfg.run
    dataset_cfg = cfg.dataset
    training_cfg = cfg.training
    optimizer_cfg = cfg.optimizer
    scheduler_cfg = cfg.scheduler
    model_cfg = cfg.model
    dataset = dataset_cfg.dataset
    model_type = model_cfg.model
    glt_topology = model_type == "glt"

    case_dir = os.path.join(CASEDIR, run_cfg.exp_name)

    #=================#
    # DATA
    #=================#

    data_root = dataset_cfg.data_root if dataset_cfg.data_root is not None else DATADIR_BASE
    resolved_dataset = resolve_dataset_name(dataset.lower())
    is_plaid = resolved_dataset in PLAID_DATASETS or resolved_dataset.startswith("plaid_")
    mesh = is_plaid or (model_type in MESH_GRAPH_MODELS)
    mesh_graph_backend = "pyg"
    ginot_use_flash_varlen = resolve_use_flash_varlen(
        model_type=model_type,
        dataset_name=dataset,
        mixed_precision=bool(training_cfg.mixed_precision),
        use_context_parallel=bool(training_cfg.use_context_parallel),
    )
    if ginot_use_flash_varlen and GLOBAL_RANK == 0:
        print(
            "Packed flash-attn varlen workflow selected (GLT, or mixed-precision FLARE/GITO on "
            "GINOT/LPBF varlen batches). No masked-SDPA fallback will be used; unsupported "
            "dtype/device/metadata will fail loudly."
        )
    if ginot_use_flash_varlen:
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Packed flash-attn varlen requires flash-attn. "
                "Install flash-attn, disable mixed_precision, or use a dense/padded path "
                "(e.g. nasa_crm / context parallel)."
            ) from exc
    ginot_pipeline = uses_ginot_pipeline(dataset, model_type)
    if glt_topology:
        pe_feature_request = model_cfg.pe.to_feature_request()
        feature_request = FeatureRequest(
            edges=pe_feature_request.edges or (model_type in EDGE_INFO_MODELS),
            boundary=pe_feature_request.boundary,
            laplacian_k=pe_feature_request.laplacian_k,
            laplacian_spec=pe_feature_request.laplacian_spec,
            pos_domain=bool(pe_feature_request.pos_domain),
        )
    else:
        feature_request = FeatureRequest(
            edges=(model_type in EDGE_INFO_MODELS),
            boundary=False,
        )
    load_kwargs = dict(
        mesh=mesh,
        ginot_use_flash_varlen=ginot_use_flash_varlen,
        model_type=model_type,
        feature_request=feature_request,
    )
    if ginot_pipeline:
        load_kwargs.update(
            mesh_split_seed=dataset_cfg.mesh_split_seed,
            ginot_max_samples=dataset_cfg.max_samples,
        )
    elif is_plaid:
        load_kwargs.update(
            mesh_split_seed=run_cfg.seed,
            plaid_use_sdf_features=dataset_cfg.plaid_use_sdf_features,
            plaid_load_public_test=dataset_cfg.plaid_load_public_test,
            plaid_terminal_y_norm=dataset_cfg.plaid_terminal_y_norm,
            plaid_terminal_target_fields=dataset_cfg.plaid_terminal_target_fields,
            plaid_max_samples=dataset_cfg.max_samples,
        )
    elif dataset in ("ahmedml_surface", "drivaerml_surface"):
        load_kwargs["subset_size"] = dataset_cfg.subset_size
        load_kwargs["iid_samples"] = dataset_cfg.iid_samples
    timer.mark("dataset_load_start")
    _data, data_, metadata = pdebench.load_dataset(
        dataset,
        data_root,
        PROJDIR,
        **load_kwargs,
    )
    timer.mark("dataset_loaded")

    if metadata is None:
        raise ValueError("metadata is None. Check pdebench.load_dataset and your dataset path/configuration.")
    metadata["dataset"] = resolved_dataset
    metadata["model"] = model_type
    if dataset == "ahmedml_surface":
        if metadata.get("ahmedml_train_run_data") is None or metadata.get("ahmedml_test_run_data") is None:
            raise ValueError("ahmedml_surface metadata missing ahmedml_train_run_data / ahmedml_test_run_data")
        metadata["rel_l2_loss"] = bool(dataset_cfg.rel_l2_loss)
    elif dataset == "drivaerml_surface":
        if metadata.get("drivaerml_train_run_data") is None or metadata.get("drivaerml_test_run_data") is None:
            raise ValueError("drivaerml_surface metadata missing drivaerml_train_run_data / drivaerml_test_run_data")
        metadata["rel_l2_loss"] = bool(dataset_cfg.rel_l2_loss)
    lpbf_graph_cache = dataset in LPBF_DATASETS and (
        bool(metadata.get("ginot_include_edges")) or model_type == "glt"
    )

    if GLOBAL_RANK == 0:
        test_count = 0 if data_ is None else len(data_)
        print(
            f"Loaded {dataset} dataset with {len(_data)} train and {test_count} test cases "
            f"(elapsed={timer.elapsed('dataset_loaded'):.3f}s)."
        )
        train_mean_nodes = metadata.get("ginot_train_mean_nodes_per_run")
        if train_mean_nodes is not None:
            print(
                f"GINOT train node counts: mean_per_run={float(train_mean_nodes):.6f}; "
                f"mean_per_batch_bs1={float(metadata['ginot_train_mean_nodes_per_batch_bs1']):.6f}."
            )

    from pdebench.dataset.pos_domain import maybe_attach_pos_domain

    if glt_topology and feature_request.pos_domain:
        timer.mark("pos_domain_scan_start")
        metadata = maybe_attach_pos_domain(
            metadata,
            _data,
            batch_size=int(training_cfg.batch_size),
            feature_request=feature_request,
            num_workers=0,
        )
        timer.mark("pos_domain_scan_done")
        if GLOBAL_RANK == 0:
            expanse = metadata["pos_domain"].normalized_pos_expanse
            print(
                f"PosDomain normalized_pos_expanse={expanse.tolist()} "
                f"(elapsed={timer.elapsed('pos_domain_scan_done'):.3f}s)."
            )

    #=================#
    # MODEL
    #=================#

    timer.mark("model_build_start")
    cfg, model = make_model(cfg, metadata, GLOBAL_RANK)
    timer.mark("model_built")

    if run_cfg.train and GLOBAL_RANK == 0:
        config_file = os.path.join(case_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            yaml.safe_dump(cfg.to_dict(), f)

    if dataset == 'navier_stokes' and training_cfg.use_context_parallel:
        raise NotImplementedError("Navier-Stokes rollout training does not yet support context parallelism.")
    if dataset == 'plasticity' and training_cfg.use_context_parallel:
        raise NotImplementedError("Plasticity time-conditioned training does not yet support context parallelism.")

    # Handle time-conditioned models
    if metadata['time_cond'] and dataset not in ['plasticity'] and not mesh_static_supports_time_cond(dataset):
        raise NotImplementedError("Time-conditioned models not implemented in this repository.")

    cp_state = None
    cp_preprocess_fn = None
    preprocess_fns = []
    if training_cfg.use_context_parallel:
        if not DISTRIBUTED:
            raise ValueError("use_context_parallel=True requires torchrun/distributed execution.")
        cp_state = build_context_parallel_state(training_cfg.context_parallel_size)
        if not hasattr(model, "set_context_parallel"):
            raise ValueError(
                f"Model type '{model_type}' does not expose set_context_parallel; "
                "CP requires a model that knows how to shard and reduce its sequence-aligned tensors."
            )
        model.set_context_parallel(
            cp_state=cp_state,
            cp_debug_gather_outputs=training_cfg.cp_debug_gather_outputs,
        )

        if GLOBAL_RANK == 0:
            print(
                f"Context parallel enabled: cp_size={cp_state.cp_size}, "
                f"global_world={cp_state.world_size}, cp_sequence_dim={training_cfg.cp_sequence_dim}."
            )

        def _cp_preprocess_fn(batch):
            return shard_batch(batch=batch, cp_state=cp_state, seq_dim=training_cfg.cp_sequence_dim)

        cp_preprocess_fn = _cp_preprocess_fn
        preprocess_fns.append(cp_preprocess_fn)

    if GLOBAL_RANK == 0:
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # compute timings over 1 epoch with batch size 1
    if run_cfg.timing_only:
        training_cfg.batch_size = 1
        training_cfg.epochs = 2

    #=================#
    # MAKE TRAINER
    #=================#

    #----------#
    # callback
    #----------#

    callback = mlutils.Callback(case_dir)
    drivaerml_1m_use_normalized_mse = (dataset == 'drivaerml_1m')

    if dataset in ['navier_stokes']:
        callback = pdebench.NavierStokesCallback(case_dir)
    elif dataset in ['plasticity']:
        callback = pdebench.PlasticityCallback(case_dir)
    elif is_plaid:
        callback = pdebench.MeshStaticCallback(
            case_dir=case_dir,
            model_type=model_type,
            y_normalizer=metadata['y_normalizer'],
            y_scalar_normalizer=metadata.get('y_scalar_normalizer'),
            target_fields=metadata.get('target_fields'),
            target_scalar_fields=metadata.get('target_scalar_fields'),
            test_data=metadata.get('mesh_test_data'),
            dataset_name=dataset,
        )
    elif dataset == 'ahmedml_surface':
        callback = pdebench.AhmedMLSurfaceRelL2Callback(
            case_dir,
            dataset,
            metadata['x_normalizer'],
            metadata['y_normalizer'],
            y_field_slices=metadata.get('y_field_slices'),
            y_field_metrics=metadata.get('y_field_metrics'),
        )
    elif dataset == 'drivaerml_surface':
        callback = pdebench.DrivAerMLSurfaceRelL2Callback(
            case_dir,
            dataset,
            metadata['x_normalizer'],
            metadata['y_normalizer'],
            y_field_slices=metadata.get('y_field_slices'),
            y_field_metrics=metadata.get('y_field_metrics'),
        )
    elif dataset in [
        'elasticity', 'darcy', 'airfoil_steady', 'pipe',
        'shapenet_car', 'nasa_crm', 'airfrans', 'am_small',
    ] or (dataset.startswith('drivaerml') and not drivaerml_1m_use_normalized_mse):
        callback = pdebench.RelL2Callback(
            case_dir,
            dataset,
            metadata['x_normalizer'],
            metadata['y_normalizer'],
            y_field_slices=metadata.get('y_field_slices'),
            y_field_metrics=metadata.get('y_field_metrics'),
        )
    elif dataset in LPBF_DATASETS and not lpbf_graph_cache:
        import am
        callback = am.FinaltimeCallback(case_dir, mesh=mesh, num_eval_cases=20)
    elif dataset in ['am_dynamic']:
        import am

        callback = am.TimeseriesCallback(case_dir, mesh=mesh, num_eval_cases=20, autoreg_start=1)

    # use scores callback in eval mode
    if model_type in ['flare', 'flare_ablations'] and run_cfg.evaluate and dataset in [
        'elasticity', 'darcy', 'airfoil_steady', 'shapenet_car', 'airfrans',
    ]:
        callback = pdebench.ScoresCallback(case_dir)

    #----------#
    # batch_size (global unless context-parallel sets batch_size_is_per_rank below)
    #----------#

    _batch_size = training_cfg.batch_size
    if ginot_pipeline:
        batch_size_ = _batch_size_ = _batch_size
    elif dataset in ['navier_stokes', 'plasticity']:
        batch_size_ = _batch_size_ = _batch_size
    elif model_type == 'transolver' and dataset in ['elasticity']:
        batch_size_ = _batch_size_ = _batch_size
    elif is_plaid:
        batch_size_ = _batch_size_ = _batch_size
    elif dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
    ]:
        batch_size_ = _batch_size_ = 5
    elif dataset in LPBF_DATASETS:
        # Multi-graph FLARE: padded+mask or flat varlen (see lpbf_batch_format).
        batch_size_ = _batch_size_ = _batch_size
        if _batch_size % WORLD_SIZE != 0:
            raise ValueError(
                f"Global batch_size={_batch_size} must be divisible by WORLD_SIZE={WORLD_SIZE} for LPBF."
            )
    else:
        batch_size_ = _batch_size_ = 1

    _graph_native_datasets = LPBF_DATASETS
    gnn_loader = (
        dataset in _graph_native_datasets
        and not ginot_pipeline
        and not metadata.get("ginot_include_edges")
    )
    graph_loader_backend = 'pyg' if gnn_loader else None
    if model_type in MESH_SEQUENCE_MODELS and dataset not in _graph_native_datasets:
        gnn_loader = False
        graph_loader_backend = None
    elif is_plaid:
        gnn_loader = model_type in MESH_GRAPH_MODELS
        graph_loader_backend = mesh_graph_backend if gnn_loader else None
        if metadata.get("sample_collate") and gnn_loader:
            # C4: the dataset yields Sample (not Data) for this family, but
            # torch_geometric.loader.DataLoader always builds its own Collater
            # and ignores any collate_fn passed to it, so a Sample-yielding
            # dataset would crash under the pyg graph loader. Route through the
            # plain torch DataLoader instead so metadata['train_collate_fn'] /
            # ['eval_collate_fn'] (collate_plaid_static) actually runs.
            gnn_loader = False
            graph_loader_backend = None

    #----------#
    # make_optimizer
    #----------#

    if optimizer_cfg.optimizer == 'adam':
        def make_optimizer(model, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
            )
    elif optimizer_cfg.optimizer == 'adamw':
        if model_type == 'transolver' and dataset in ['elasticity', 'navier_stokes', 'plasticity']:
            make_optimizer = pdebench.make_optimizer_plain_adamw
        else:
            make_optimizer = pdebench.make_optimizer_adamw
            c_stream_wd = optimizer_cfg.glt_c_stream_weight_decay
            if (
                c_stream_wd is not None
                and model_type == 'glt'
                and bool(getattr(model_cfg, 'pe_update', False))
            ):
                base_make_optimizer = make_optimizer

                def make_optimizer(model, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
                    return base_make_optimizer(
                        model,
                        lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        c_stream_weight_decay=float(c_stream_wd),
                    )

                if GLOBAL_RANK == 0:
                    print(
                        f"GLT dual-stream optimizer: weight_decay={optimizer_cfg.weight_decay} (x-stream), "
                        f"glt_c_stream_weight_decay={c_stream_wd} (c_proj + blocks_c)"
                    )
    elif optimizer_cfg.optimizer == 'lion':
        make_optimizer = pdebench.make_optimizer_lion
    elif optimizer_cfg.optimizer == 'muon':
        scheduler_cfg.cycle_momentum = False
        make_optimizer = pdebench.make_optimizer_muon
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_cfg.optimizer}. Choose from adamw, lion, muon.")

    #----------#
    # lossfun
    #----------#

    if ginot_pipeline:
        if GLOBAL_RANK == 0:
            print(f"Using per-channel relative L2 objective and stats for {dataset} GINOT dataset")
        lossfun = None
    elif is_plaid:
        if GLOBAL_RANK == 0:
            lbda = float(metadata.get("plaid_loss_lbda", 0.5))
            scalars = metadata.get("target_scalar_fields", [])
            print(
                f"Using Vi-Transf λ-blended MSE objective and stats for {dataset} "
                f"(lbda={lbda}, scalars={list(scalars)})"
            )
        lossfun = None
    elif dataset in MSE_NORMALIZED_DATASETS:
        # nasa_crm: MSE on normalized targets.
        # ahmedml/drivaerml surfaces: MSE by default; optional physical Rel-L2 blend.
        if dataset in ("ahmedml_surface", "drivaerml_surface") and bool(metadata.get("rel_l2_loss", False)):
            if GLOBAL_RANK == 0:
                print(
                    f"Using 0.5*(pressure Rel-L2 + wall-shear Rel-L2) for {dataset} "
                    "(physical units; per-batch and full-mesh)"
                )

            def lossfun(yh, y):
                return surface_batch_loss(
                    yh,
                    y,
                    rel_l2_loss=True,
                    y_normalizer=metadata["y_normalizer"],
                    cp_state=None,
                )
        else:
            if GLOBAL_RANK == 0:
                print(f"Using MSELoss (normalized) for {dataset} dataset")
            lossfun = torch.nn.MSELoss()
    elif (
        (dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
        'shapenet_car',
        ] or dataset.startswith('drivaerml'))
        and not drivaerml_1m_use_normalized_mse
    ):
        if GLOBAL_RANK == 0:
            print(f"Using RelL2Loss for {dataset} dataset")
        lf = pdebench.RelL2Loss()
        def lossfun(yh, y):
            y_normalizer = metadata['y_normalizer'].to(y.device)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            return lf(yh, y)
    elif dataset in LPBF_DATASETS and lpbf_graph_cache:
        if GLOBAL_RANK == 0:
            print(f"Using per-channel relative L2 objective and stats for {dataset} LPBF graph-cache path")
        lossfun = None
    elif dataset in LPBF_DATASETS:
        lpbf_batch_format = resolve_lpbf_batch_format(
            mixed_precision=bool(training_cfg.mixed_precision),
            explicit=metadata.get("lpbf_batch_format"),
            use_context_parallel=bool(training_cfg.use_context_parallel),
        )
        metadata["lpbf_batch_format"] = lpbf_batch_format
        lpbf_collate = make_lpbf_collate_fn(lpbf_batch_format)
        metadata["train_collate_fn"] = lpbf_collate
        metadata["eval_collate_fn"] = lpbf_collate
        if GLOBAL_RANK == 0:
            print(
                f"Using LPBF physical channel-mean Rel-L2 ({lpbf_batch_format} batch format) "
                f"for {dataset} dataset"
            )
        def lossfun(yh, y):
            return lpbf_warped_rel_l2(yh, y, metadata['y_normalizer'])
    else:
        if GLOBAL_RANK == 0:
            print(f"Using MSELoss for {dataset} dataset")
        lossfun = torch.nn.MSELoss()

    #----------#
    # Trainer kwargs
    #----------#

    clip_grad_norm = training_cfg.clip_grad_norm
    if model_type in {
        "meshgraphnet",
        "rigno",
        "gito",
        "geo_transolver",
    }:
        # Match PLAID MGN training loop (no gradient clipping).
        clip_grad_norm = None

    kw = dict(
        # device & compilation
        device=device, mixed_precision=training_cfg.mixed_precision, amp_dtype=training_cfg.amp_dtype,
        compile_model=training_cfg.compile_model,
        static_graph=training_cfg.static_graph,
        ddp_find_unused_params=(model_type == "glt" and model_cfg.pe_inject_mode == "concat_input"),
        ddp_gradient_as_bucket_view=True,
        ema=training_cfg.ema, ema_decay=training_cfg.ema_decay,
        # batch size
        _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
        # optimizer
        make_optimizer=make_optimizer,
        weight_decay=optimizer_cfg.weight_decay,
        epochs=training_cfg.epochs,
        steps=training_cfg.steps,
        lossfun=lossfun, clip_grad_norm=clip_grad_norm,
        opt_beta1=optimizer_cfg.opt_beta1, opt_beta2=optimizer_cfg.opt_beta2, opt_eps=optimizer_cfg.opt_eps,
        # dataloader kwargs
        num_workers=training_cfg.num_workers, prefetch_factor=training_cfg.prefetch_factor,
        overlap_train_dataloader=training_cfg.overlap_train_dataloader,
        continuous_train_batches=training_cfg.continuous_train_batches,
        gnn_loader=gnn_loader, graph_loader_backend=graph_loader_backend,
        # stats controls
        _fullbatch_stats=training_cfg.fullbatch_stats_train,
        fullbatch_stats_=training_cfg.fullbatch_stats_test,
        fullbatch_stats_on_start=training_cfg.fullbatch_stats_on_start,
        stats_on_start=training_cfg.stats_on_start,
    )
    # Keep training compiled when requested, but run full-batch stats eagerly for
    # variable-length GINOT packs and full-mesh surface Rel-L2 (shape ≠ train 100k).
    kw['compile_stats_model'] = compile_stats_model_for_dataset(dataset, model_type, training_cfg.compile_model)
    if training_cfg.compile_model and not kw['compile_stats_model'] and GLOBAL_RANK == 0:
        print("Using eager model for full-batch stats (variable-length / full-mesh eval).")
    if dataset == 'navier_stokes':
        kw['statsfun'] = make_navier_stokes_statsfun()
    elif dataset == 'plasticity':
        kw['statsfun'] = make_plasticity_statsfun()
    elif dataset == 'ahmedml_surface':
        kw['statsfun'] = make_ahmedml_surface_statsfun(metadata, cp_state=cp_state)
    elif dataset == 'drivaerml_surface':
        kw['statsfun'] = make_drivaerml_surface_statsfun(metadata, cp_state=cp_state)
    elif is_plaid:
        kw['statsfun'] = make_mesh_static_statsfun(cfg, metadata)
    elif lpbf_graph_cache:
        kw['statsfun'] = make_lpbf_statsfun(cfg, metadata)
    elif ginot_pipeline:
        kw['statsfun'] = make_ginot_statsfun(cfg, metadata)
    if metadata.get('train_collate_fn') is not None:
        kw['_collate_fn'] = metadata['train_collate_fn']
    if is_plaid and model_type in MESH_SEQUENCE_MODELS:
        kw['_collate_fn'] = mesh_sequence_collate_fn
    if dataset == 'plasticity':
        kw['schedule_step_multiplier'] = metadata.get('rollout_steps', 1)
    if metadata.get('eval_collate_fn') is not None:
        kw['collate_fn_'] = metadata['eval_collate_fn']
    if is_plaid and model_type in MESH_SEQUENCE_MODELS:
        kw['collate_fn_'] = mesh_sequence_collate_fn
    if preprocess_fns:
        def _combined_preprocess_fn(batch):
            for preprocess_fn in preprocess_fns:
                batch = preprocess_fn(batch)
            return batch

        kw["_preprocess_fn"] = _combined_preprocess_fn
        kw["preprocess_fn_"] = _combined_preprocess_fn
    if cp_preprocess_fn is not None:
        kw["batch_size_is_per_rank"] = True
        kw["use_distributed_sampler"] = False
    if GLOBAL_RANK == 0:
        if kw.get("batch_size_is_per_rank"):
            print(
                f"Training batch_size={_batch_size} (per_rank; context parallel, "
                f"WORLD_SIZE={WORLD_SIZE})"
            )
        elif DISTRIBUTED:
            print(
                f"Training batch_size: global={_batch_size} per_rank={_batch_size // WORLD_SIZE} "
                f"(WORLD_SIZE={WORLD_SIZE})"
            )
        else:
            print(f"Training batch_size: global={_batch_size} (single process)")
    if training_cfg.stats_every > 0:
        kw['stats_every'] = training_cfg.stats_every

    #----------#
    # LR scheduler
    #----------#

    if scheduler_cfg.schedule is None or scheduler_cfg.schedule == 'ConstantLR':
        kw['lr'] = optimizer_cfg.learning_rate
    elif scheduler_cfg.schedule == 'OneCycleLR':

        if scheduler_cfg.override_min_lr is not None:
            scheduler_cfg.div_factor = optimizer_cfg.learning_rate / scheduler_cfg.override_min_lr
            scheduler_cfg.final_div_factor = 1.0

        kw['Schedule'] = 'OneCycleLR'
        kw['lr'] = optimizer_cfg.learning_rate
        kw['one_cycle_pct_start'] = scheduler_cfg.pct_start
        kw['one_cycle_div_factor'] = scheduler_cfg.div_factor
        kw['one_cycle_final_div_factor'] = scheduler_cfg.final_div_factor
        kw['one_cycle_three_phase'] = scheduler_cfg.three_phase
        kw['one_cycle_cycle_momentum'] = scheduler_cfg.cycle_momentum
        kw['one_cycle_base_momentum'] = scheduler_cfg.base_momentum
        kw['one_cycle_max_momentum'] = scheduler_cfg.max_momentum
        kw['one_cycle_anneal_strategy'] = scheduler_cfg.anneal_strategy
    else:
        kw = dict(**kw, Schedule=scheduler_cfg.schedule, lr=optimizer_cfg.learning_rate,)
        if scheduler_cfg.schedule == 'ReduceLROnPlateau':
            kw['plateau_factor'] = scheduler_cfg.plateau_factor
            kw['plateau_patience'] = scheduler_cfg.plateau_patience
        if scheduler_cfg.schedule == 'CosineAnnealingLR':
            kw['min_lr'] = float(getattr(scheduler_cfg, 'min_lr', 0.0) or 0.0)

    #-------------#
    # make Trainer
    #-------------#

    timer.mark("trainer_construct_start")
    kw["run_timer"] = timer
    trainer = mlutils.Trainer(model, _data, data_, **kw)
    timer.mark("trainer_constructed")

    # Match PhysicsNeMo crash recipe: CosineAnnealingLR stepped once per epoch.
    if (
        scheduler_cfg.schedule == "CosineAnnealingLR"
        and trainer.train_based_on_epochs
        and int(getattr(trainer, "epochs", 0) or 0) > 0
    ):
        trainer.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.opt,
            T_max=max(1, int(trainer.epochs)),
            eta_min=float(getattr(scheduler_cfg, "min_lr", 0.0) or 0.0),
        )
        trainer.update_schedule_every_epoch = True
    elif model_type == "transolver" and dataset == "elasticity":
        trainer.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.opt,
            T_max=trainer.epochs,
            eta_min=0.0,
        )
        trainer.update_schedule_every_epoch = True

    #-------------#
    # add callback
    #-------------#
    if trainer.train_based_on_epochs:
        trainer.add_callback('epoch_end', callback)
    else:
        trainer.add_callback('batch_end', callback)

    if model_type == 'mixer_backbone' and bool(getattr(model_cfg, 'diagnostics', False)):
        trainer.add_callback('batch_end', pdebench.MixerDiagnosticsCallback(case_dir))

    #-------------#
    # batch_lossfun
    #-------------#
    if training_cfg.use_context_parallel:
        # nasa_crm and surface datasets use MSE on normalized targets (falls through to cp_reduced_mse_loss).
        rel_l2_datasets = {
            'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
            'shapenet_car',
        }

        def batch_lossfun(trainer, model, batch):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("CP training currently expects batch format (x, y, ...).")
            x, y = batch[0], batch[1]
            yh = model(x)

            if dataset in ("ahmedml_surface", "drivaerml_surface"):
                return surface_batch_loss(
                    yh,
                    y,
                    rel_l2_loss=bool(metadata.get("rel_l2_loss", False)),
                    y_normalizer=metadata["y_normalizer"],
                    cp_state=cp_state,
                )
            if dataset in MSE_NORMALIZED_DATASETS:
                return cp_reduced_mse_loss(yh, y, cp_state=cp_state)
            if (
                ((dataset in rel_l2_datasets) or dataset.startswith('drivaerml'))
                and not drivaerml_1m_use_normalized_mse
            ):
                y_normalizer = metadata['y_normalizer'].to(y.device)
                yh = y_normalizer.decode(yh)
                y = y_normalizer.decode(y)
                return cp_reduced_rel_l2_loss(yh, y, cp_state=cp_state)

            return cp_reduced_mse_loss(yh, y, cp_state=cp_state)

        trainer.batch_lossfun = batch_lossfun

    elif is_plaid:
        def batch_lossfun(trainer, model, batch):
            yh, y, batch_index, num_graphs = mesh_model_forward(cfg, model, batch)
            if y is None:
                raise ValueError("Received unlabeled mesh batch during training; cannot compute PLAID loss.")
            del trainer
            field_dim = len(metadata.get("target_fields", []))
            scalar_dim = len(metadata.get("target_scalar_fields", []))
            lbda = float(metadata.get("plaid_loss_lbda", 1.0 if scalar_dim == 0 else 0.5))
            loss, _field_mse, _scalar_mse = mesh_batch_plaid_scaled_mse(
                yh,
                y,
                batch,
                batch_index=batch_index,
                num_graphs=num_graphs,
                field_dim=field_dim,
                scalar_dim=scalar_dim,
                lbda=lbda,
            )
            return loss

        trainer.batch_lossfun = batch_lossfun

    elif lpbf_graph_cache:
        def batch_lossfun(trainer, model, batch):
            yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)
            y_normalizer = metadata['y_normalizer'].to(y.device)
            return lpbf_warped_rel_l2(
                yh,
                y,
                y_normalizer,
                batch_index=batch_index,
                num_graphs=num_graphs,
            )

        trainer.batch_lossfun = batch_lossfun

    elif ginot_pipeline:
        def batch_lossfun(trainer, model, batch):
            yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)
            y_normalizer = metadata['y_normalizer'].to(y.device)
            yh = ginot_postprocess_displacement(yh, batch, y_normalizer=y_normalizer)
            loss_spec = LossSpec(mask="free_mask" if batch.get("flat_free_mask") is not None else None)
            masks = {"free_mask": batch["flat_free_mask"]} if loss_spec.mask else None
            return compute_packed_loss(
                yh,
                y,
                y_normalizer,
                loss_spec,
                batch_index=batch_index,
                num_graphs=num_graphs,
                masks=masks,
            )

        trainer.batch_lossfun = batch_lossfun

    elif dataset in ['darcy']:

        r = 5
        h = int(((421 - 1) / r) + 1)
        s = h
        dx = 1.0 / s
        lf = pdebench.RelL2Loss()

        def batch_lossfun(trainer, model, batch):
            x, y = batch
            yh = model(x)

            y_normalizer = metadata['y_normalizer'].to(y.device)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)

            l2 = lf(yh, y)
            (gt_grad_x, gt_grad_y), (pred_grad_x, pred_grad_y) = pdebench.darcy_deriv_loss(yh, y, s, dx)
            deriv_loss = lf(pred_grad_x, gt_grad_x) + lf(pred_grad_y, gt_grad_y)

            loss = 0.1 * deriv_loss + l2
            # loss = l2
            return loss

        trainer.batch_lossfun = batch_lossfun

    elif dataset in LPBF_DATASETS:

        def batch_lossfun(trainer, model, batch):
            return lpbf_flare_batch_loss(
                model,
                batch,
                metadata["y_normalizer"],
            )

        trainer.batch_lossfun = batch_lossfun

    elif dataset in ['navier_stokes']:

        lf = pdebench.RelL2Loss()

        def batch_lossfun(trainer, model, batch):
            pos, history, target = batch
            _, step_loss, _ = pdebench.rollout_navier_stokes(
                model,
                pos,
                history,
                target,
                lossfun=lf,
                teacher_forcing=True,
            )
            return step_loss

        trainer.batch_lossfun = batch_lossfun

    elif dataset in ['plasticity']:

        lf = pdebench.RelL2Loss()

        def batch_lossfun(trainer, model, batch):
            pos, time_grid, features, target = batch
            _, step_loss, _ = pdebench.rollout_plasticity(
                model,
                pos,
                time_grid,
                features,
                target,
                lossfun=lf,
            )
            return step_loss

        trainer.batch_lossfun = batch_lossfun

        def plasticity_train_step(batch):
            if trainer.grad_accumulation_steps != 1:
                raise NotImplementedError("Plasticity upstream training parity requires grad_accumulation_steps == 1.")

            if trainer.is_cuda:
                torch.cuda.reset_peak_memory_stats()

            batch_start_time = time.time()
            trainer.model.train()

            batch = trainer.move_to_device(batch)
            batch = trainer.apply_preprocessor(batch, split='train')

            pos, time_grid, features, target = batch
            bsz = pos.shape[0]
            num_steps = target.shape[-1]

            trainer.opt.zero_grad()

            model_eval_start = time.time()
            step_loss_total = 0.0
            grad_norm = float('nan')

            for t in range(num_steps):
                current_target = target[..., t:t + 1]
                current_time = time_grid[:, t:t + 1].reshape(bsz, 1)

                with trainer.auto_cast:
                    pred = trainer.model(pos, features, current_time)
                    loss = lf(pred.reshape(bsz, -1), current_target.reshape(bsz, -1))

                step_loss_total += loss.item()
                trainer.grad_scaler.scale(loss).backward()
                trainer.trigger_callbacks("batch_post_grad")
                trainer.grad_scaler.unscale_(trainer.opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_grad_norm).item()
                trainer.grad_scaler.step(trainer.opt)
                trainer.grad_scaler.update()
                trainer.opt.zero_grad()

                if not trainer.update_schedule_every_epoch:
                    trainer.schedule.step()

                if trainer.use_ema:
                    trainer.ema.update(trainer.model)

            model_eval_end = time.time()
            trainer.time_model_eval_per_step.append(model_eval_end - model_eval_start)

            trainer.train_loss_per_batch.append(step_loss_total)
            trainer.grad_norm_per_step.append(grad_norm)
            for (i, lr) in enumerate(trainer.schedule.get_last_lr()):
                trainer.learning_rates_per_step[i].append(lr)

            trainer.time_per_step.append(time.time() - batch_start_time)

            if trainer.is_cuda:
                trainer.record_cuda_memory()

            return torch.tensor(step_loss_total, device=trainer.device)

        trainer.train_step = plasticity_train_step

    #-------------#
    # load snapshot
    #-------------#

    if run_cfg.restart:
        callback.load_latest_checkpoint(trainer)
    if run_cfg.load_weights_path is not None:
        trainer.load_weights(run_cfg.load_weights_path)

    #=================#
    # TRAIN
    #=================#

    if run_cfg.train and (training_cfg.epochs > 0 or training_cfg.steps > 0):
        timer.mark("train_start")
        trainer.train()
        timer.mark("train_done")

    #=================#
    # ANALYSIS
    #=================#

    if run_cfg.evaluate:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer.make_dataloader()
        callback.load_latest_checkpoint(trainer)
        trainer.statistics()
        callback(trainer, final=True)

    timer.mark("main_exit")
    return

#======================================================================#
if __name__ == "__main__":

    argv = list(sys.argv)
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    _forced_device = os.environ.get('PDEBENCH_DEVICE')
    if _forced_device is not None:
        device = torch.device(_forced_device)
    else:
        device = mlutils.select_device()

    run_timer = mlutils.RunTimer(rank=GLOBAL_RANK, log_rank=0)
    run_timer.mark("process_start")

    #===============#
    parser = ArgumentParser()
    parser.add_class_arguments(Config, nested_key=None)
    parsed = parser.parse_args()
    cfg = Config(**parsed.as_dict())
    #===============#

    if (cfg.run.train + cfg.run.evaluate + cfg.run.restart) != 1:
        msg = (
            "Invalid mode selection. Select one of "
            f"train (got {cfg.run.train}), evaluate (got {cfg.run.evaluate}), restart (got {cfg.run.restart})."
        )
        raise ValueError(msg)

    cli_cfg_dict = cfg.to_dict()

    case_dir: str | None = None
    log_path: str | None = None

    if cfg.run.train:
        cfg.run.exp_name = mlutils.get_next_exp_name(CASEDIR, cfg.run.exp_name)
        case_dir = os.path.join(CASEDIR, cfg.run.exp_name)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if GLOBAL_RANK == 0:
            os.makedirs(case_dir, exist_ok=True)
            log_path = os.path.join(case_dir, "log.txt")
            mlutils.setup_run_log(log_path, rank=GLOBAL_RANK)
            mlutils.log_run_banner(
                argv=argv,
                cfg_dict=cli_cfg_dict,
                case_dir=case_dir,
                log_path=log_path,
            )
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(cfg.to_dict(), f)
            mlutils.log_config_trace(argv=argv, cfg_dict=cfg.to_dict(), title="resolved config (train)")

    # load config from experiment directory
    if cfg.run.evaluate or cfg.run.restart:
        case_dir = os.path.join(CASEDIR, cfg.run.exp_name)
        assert os.path.exists(case_dir), f"Experiment directory {case_dir} does not exist."
        config_file = os.path.join(case_dir, 'config.yaml')

        # save original config
        _cfg = cfg

        if GLOBAL_RANK == 0:
            log_path = os.path.join(case_dir, "log.txt")
            mlutils.setup_run_log(log_path, rank=GLOBAL_RANK)
            mlutils.log_run_banner(
                argv=argv,
                cfg_dict=cli_cfg_dict,
                case_dir=case_dir,
                log_path=log_path,
            )
            print(f'Loading config from {config_file}')

        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = Config(**cfg)

        if _cfg.run.evaluate:
            cfg.run.evaluate = True
            cfg.run.train = False
        elif _cfg.run.restart:
            cfg.run.restart = True
            cfg.run.train = True

        if GLOBAL_RANK == 0:
            mlutils.log_config_trace(argv=argv, cfg_dict=cfg.to_dict(), title="resolved config (evaluate/restart)")

    # after evaluate/restart config reload
    #===============#
    runtime = mlutils.configure_runtime(
        cfg.run.seed,
        mixed_precision=bool(cfg.training.mixed_precision),
        deterministic=bool(cfg.run.deterministic),
        compile_model=bool(cfg.training.compile_model),
    )
    if GLOBAL_RANK == 0:
        print(
            "runtime_profile={profile} seed={seed} tf32={tf32} "
            "cudnn.benchmark={cudnn_benchmark} cudnn.deterministic={cudnn_deterministic} "
            "deterministic_algorithms={deterministic_algorithms}".format(**runtime)
        )
    #===============#

    if DISTRIBUTED:
        torch.distributed.barrier()

    try:
        main(cfg, device, run_timer=run_timer)
        run_timer.mark("process_done")
        run_timer.print_summary()
    finally:
        mlutils.close_run_log()

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#
