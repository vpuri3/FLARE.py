#
import torch

import os
import yaml
from jsonargparse import CLI
from dataclasses import dataclass

# local
import am
import pdebench
import mlutils

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'pdebench')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti 11 GB
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')

os.environ["HF_HOME"] = os.path.join(DATADIR_BASE, 'huggingface')

#======================================================================#
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    mesh = cfg.model_type in [-1,]
    _data, data_, metadata = pdebench.load_dataset(cfg.dataset, DATADIR_BASE, PROJDIR, mesh=mesh)

    if metadata is None:
        raise ValueError("metadata is None. Check pdebench.load_dataset and your dataset path/configuration.")

    c_in = metadata['c_in']
    c_edge = metadata.get('c_edge', None)
    c_out = metadata['c_out']

    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")
        # print(f"Number of points: {len(next(_data))}")

    #=================#
    # MODEL
    #=================#

    if metadata['time_cond']:
        raise NotImplementedError("Time-conditioned models not implemented in this repository.")

        # Use masked model for timeseries datasets
        if cfg.dataset in ['airfoil', 'cylinder_flow']:
            if GLOBAL_RANK == 0:
                print(f"Using masked model for timeseries datasets {cfg.dataset}")
            model = am.MaskedModel(model, mask=True, reduce_graph=False)

    else:
        if cfg.model_type == 0:

            #--------------------------------#
            # Transolver
            #--------------------------------#

            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            elif cfg.dataset in ['airfoil_steady', 'pipe', 'darcy', 'airfoil_dynamic', 'cylinder_flow']:
                cfg.batch_size = 4
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.3
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 0.1
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            n_layers = 8
            n_hidden = 128 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 256
            slice_num = 64 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 32
            n_head = 8
            mlp_ratio = 1.0

            if GLOBAL_RANK == 0:

                print(f"Using Transolver(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tn_layers={n_layers}\n"
                    + f"\tn_hidden={n_hidden}\n"
                    + f"\tslice_num={slice_num}\n"
                    + f"\tn_head={n_head}\n"
                    + f"\tmlp_ratio={mlp_ratio}\n"
                    + f"\tconv2d={cfg.conv2d}\n"
                    + f"\tunified_pos={cfg.unified_pos}\n"
                )

            if cfg.conv2d:
                model = pdebench.Transolver_Structured_Mesh_2D(
                    space_dim=c_in, out_dim=c_out, fun_dim=0,
                    n_hidden=n_hidden, n_layers=n_layers,
                    n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
                    H=metadata['H'], W=metadata['W'],
                    unified_pos=cfg.unified_pos,
                )
            else:
                model = pdebench.Transolver(
                    space_dim=c_in, out_dim=c_out, fun_dim=0,
                    n_hidden=n_hidden, n_layers=n_layers,
                    n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
                )
        elif cfg.model_type == 1:

            #--------------------------------#
            # LNO: https://github.com/L-I-M-I-T/LatentNeuralOperator
            #--------------------------------#

            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 4
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.99
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.2
            cfg.one_cycle_div_factor = 1e4
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 1000.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 5e-5

            n_head = 8
            n_mode = 256
            n_dim = 192 if cfg.dataset in ['elasticity'] else 128
            n_layer = 3 if cfg.dataset in ['elasticity'] else 2
            n_block = 8 if cfg.dataset in ['pipe', 'airfoil_steady'] else 4

            if GLOBAL_RANK == 0:
                print(f"Using LNO(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tnum_block={n_block}\n"
                    + f"\tnum_modes={n_mode}\n"
                    + f"\tchannel_dim={n_dim}\n"
                    + f"\tnum_heads={n_head}\n"
                    + f"\tnum_residual_layers={n_layer}\n"
                )

            model = pdebench.LNO(
                n_block=n_block, n_mode=n_mode, n_dim=n_dim, n_head=n_head, n_layer=n_layer, act="GELU",
                x_dim=c_in, y1_dim=c_in, y2_dim=c_out,
                model_attr={"time": metadata['time_cond'],}
            )

        elif cfg.model_type == 2:

            #--------------------------------#
            # FLARE
            #--------------------------------#

            if GLOBAL_RANK == 0:
                print(
                    f"Using FLAREModel(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tchannel_dim={cfg.channel_dim}\n"
                    + f"\tnum_blocks={cfg.num_blocks}\n"
                    + f"\tnum_latents={cfg.num_latents}\n"
                    + f"\tnum_heads={cfg.num_heads}\n"
                    + f"\tact={cfg.act}\n"
                    + f"\tnum_layers_kv_proj={cfg.num_layers_kv_proj}\n"
                    + f"\tnum_layers_mlp={cfg.num_layers_mlp}\n"
                    + f"\tnum_layers_in_out_proj={cfg.num_layers_in_out_proj}\n"
                    + f"\tmlp_ratio={cfg.mlp_ratio}\n"
                    + f"\tkv_proj_ratio={cfg.kv_proj_ratio}\n"
                    + f"\tin_out_proj_ratio={cfg.in_out_proj_ratio}\n"
                    + f"\tout_proj_ln={cfg.out_proj_ln}\n"
                )

            model = pdebench.FLAREModel(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=cfg.channel_dim,
                num_blocks=cfg.num_blocks,
                num_latents=cfg.num_latents,
                num_heads=cfg.num_heads,
                act=cfg.act,
                num_layers_kv_proj=cfg.num_layers_kv_proj,
                num_layers_mlp=cfg.num_layers_mlp,
                num_layers_in_out_proj=cfg.num_layers_in_out_proj,
                mlp_ratio=cfg.mlp_ratio,
                kv_proj_ratio=cfg.kv_proj_ratio,
                in_out_proj_ratio=cfg.in_out_proj_ratio,
                out_proj_ln=cfg.out_proj_ln,
            )
        elif cfg.model_type == 3:

            #--------------------------------#
            # Vanilla Transformer
            #--------------------------------#

            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 2
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            
            # training params
            
            cfg.optimizer = 'adamw'
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.opt_eps = 1e-6 if cfg.dataset in ['pipe'] else 1e-8
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.1
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 1.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5
            
            # ADAM

            # cfg.epochs = 500
            # cfg.optimizer = 'adamw'
            # cfg.learning_rate = 1e-3
            # cfg.opt_beta1 = 0.9
            # cfg.opt_beta2 = 0.999
            # cfg.opt_eps = 1e-6
            # cfg.schedule = 'OneCycleLR'
            # cfg.one_cycle_pct_start = 0.1
            # cfg.one_cycle_div_factor = 25
            # cfg.one_cycle_final_div_factor = 1e4
            # cfg.clip_grad_norm = 1.0
            # cfg.weight_decay = 1e-5

            # LION

            # cfg.epochs = 500
            # cfg.optimizer = 'lion'
            # cfg.learning_rate = 1e-3
            # cfg.schedule = 'OneCycleLR'
            # cfg.one_cycle_pct_start = 0.1
            # cfg.one_cycle_div_factor = 25
            # cfg.one_cycle_final_div_factor = 1e4
            # cfg.clip_grad_norm = 1.0
            # cfg.weight_decay = 0e-4

            ###
            # model params
            ###

            # C = 80 (660k params), C = 96 (949k params), C = 128 (1.68m params)

            channel_dim = 80
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None

            if GLOBAL_RANK == 0:
                print(
                    f"Using Transformer(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tchannel_dim={channel_dim}\n"
                    + f"\tnum_blocks={num_blocks}\n"
                    + f"\tnum_heads={num_heads}\n"
                    + f"\tact={act}\n"
                )

            model = pdebench.Transformer(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=channel_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act=act,
            )

        elif cfg.model_type == 4:

            #--------------------------------#
            # GNOT
            #--------------------------------#

            # Learning params
            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity']:
                cfg.batch_size = 2
            elif cfg.dataset in ['darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 4
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.3
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 0.1
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['darcy']:
                cfg.weight_decay = 5e-5
            else:
                cfg.weight_decay = 1e-5

            # GNOT params
            n_layers = 8
            n_hidden = 128
            mlp_ratio = 2.0
            n_experts = 3
            n_head = 8
            if cfg.dataset in ['darcy', 'airfoil_steady', 'pipe']:
                geotype = 'structured_2D'
                unified_pos = True
                ref = 8
                shapelist = [metadata['H'], metadata['W']]
            elif cfg.dataset in ['elasticity', 'shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                geotype = 'unstructured'
                unified_pos = False
                ref = 8
                shapelist = None
            else:
                raise ValueError(f"Geotype not set for dataset {cfg.dataset}")

            if GLOBAL_RANK == 0:
                print(f"Using GNOT(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tn_experts={n_experts}\n"
                    + f"\tn_heads={n_head}\n"
                    + f"\tn_hidden={n_hidden}\n"
                    + f"\tn_layers={n_layers}\n"
                    + f"\tmlp_ratio={mlp_ratio}\n"
                    + f"\tunified_pos={unified_pos}\n"
                    + f"\tgeotype={geotype}\n"
                    + f"\tref={ref}\n"
                )

            model = pdebench.GNOT(
                n_experts=n_experts,
                n_heads=n_head,
                n_hidden=n_hidden,
                n_layers=n_layers,
                mlp_ratio=mlp_ratio,
                unified_pos=unified_pos,
                geotype=geotype,
                shapelist=shapelist,
                ref=ref,
                space_dim=c_in,
                fun_dim=0,
                out_dim=c_out,
            )
        elif cfg.model_type == 5:

            #--------------------------------#
            # UPT (Universal Physics Transformer)
            #--------------------------------#
            
            raise NotImplementedError("UPT is not implemented yet.")
            
            # Learning params
            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 2
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                cfg.batch_size = 2
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.3
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 1.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            # UPT params
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                space_dim = 2
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                space_dim = 3
            else:
                raise ValueError(f"Space dim not set for dataset {cfg.dataset}")

            d_model = 128
            n_encoder_layers = 4
            n_approximator_layers = 4
            n_decoder_layers = 2
            n_heads = 8
            d_ff = 1024
            dropout = 0.1
            use_inverse_tasks = True

            if GLOBAL_RANK == 0:
                print(f"Using UPT(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\td_model={d_model}\n"
                    + f"\tn_encoder_layers={n_encoder_layers}\n"
                    + f"\tn_approximator_layers={n_approximator_layers}\n"
                    + f"\tn_decoder_layers={n_decoder_layers}\n"
                    + f"\tn_heads={n_heads}\n"
                    + f"\td_ff={d_ff}\n"
                    + f"\tspace_dim={c_in}\n"
                    + f"\tdropout={dropout}\n"
                    + f"\tuse_inverse_tasks={use_inverse_tasks}\n"
                )

            model = pdebench.UPT(
                input_dim=c_in,
                output_dim=c_out,
                space_dim=space_dim,
                d_model=d_model,
                n_encoder_layers=n_encoder_layers,
                n_approximator_layers=n_approximator_layers,
                n_decoder_layers=n_decoder_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_inverse_tasks=use_inverse_tasks,
            )
        elif cfg.model_type == 6:

            #--------------------------------#
            # PerceiverIO
            #--------------------------------#

            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 2
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            cfg.learning_rate = 5e-4 if cfg.dataset not in ['elasticity', 'drivaerml_40k'] else 2e-4
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.1
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.clip_grad_norm = 1.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            channel_dim = 128
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None
            num_latents = 512

            cross_attn = cfg.pcvr_cross_attn

            if GLOBAL_RANK == 0:
                print(
                    f"Using PerceiverIO(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tchannel_dim={channel_dim}\n"
                    + f"\tnum_blocks={num_blocks}\n"
                    + f"\tnum_heads={num_heads}\n"
                    + f"\tmlp_ratio={mlp_ratio}\n"
                    + f"\tnum_latents={num_latents}\n"
                    + f"\tact={act}\n"
                    + f"\tcross_attn={cross_attn}\n"
                )

            model = pdebench.PerceiverIO(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=channel_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_latents=num_latents,
                act=act,
                cross_attn=cross_attn,
            )
        elif cfg.model_type == 7:

            #--------------------------------#
            # BigFLARE (ablations)
            #--------------------------------#

            if GLOBAL_RANK == 0:
                print(
                    f"Using FLAREModel(c_in={c_in}, c_out={c_out}) with\n"
                    + f"\tchannel_dim={cfg.channel_dim}\n"
                    + f"\tnum_blocks={cfg.num_blocks}\n"
                    + f"\tnum_latents={cfg.num_latents}\n"
                    + f"\tnum_heads={cfg.num_heads}\n"
                    + f"\tact={cfg.act}\n"
                    + f"\tnum_layers_kv_proj={cfg.num_layers_kv_proj}\n"
                    + f"\tnum_layers_mlp={cfg.num_layers_mlp}\n"
                    + f"\tnum_layers_in_out_proj={cfg.num_layers_in_out_proj}\n"
                    + f"\tmlp_ratio={cfg.mlp_ratio}\n"
                    + f"\tkv_proj_ratio={cfg.kv_proj_ratio}\n"
                    + f"\tin_out_proj_ratio={cfg.in_out_proj_ratio}\n"
                    + f"\tout_proj_ln={cfg.out_proj_ln}\n"
                    # ablations
                    + f"\tshared_latents={cfg.shared_latents}\n"
                    + f"\tnum_latent_blocks={cfg.num_latent_blocks}\n"
                )

            from pdebench.models.flare_ablations import BigFLAREModel
            model = BigFLAREModel(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=cfg.channel_dim,
                num_blocks=cfg.num_blocks,
                num_latents=cfg.num_latents,
                num_heads=cfg.num_heads,
                act=cfg.act,
                num_layers_kv_proj=cfg.num_layers_kv_proj,
                num_layers_mlp=cfg.num_layers_mlp,
                num_layers_in_out_proj=cfg.num_layers_in_out_proj,
                mlp_ratio=cfg.mlp_ratio,
                kv_proj_ratio=cfg.kv_proj_ratio,
                in_out_proj_ratio=cfg.in_out_proj_ratio,
                out_proj_ln=cfg.out_proj_ln,
                # ablations
                shared_latents=cfg.shared_latents,
                num_latent_blocks=cfg.num_latent_blocks,
            )
        else:
            #--------------------------------#
            # No model selected
            #--------------------------------#
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented.")

    if GLOBAL_RANK == 0:
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # compute timings over 1 epoch with batch size 1
    if cfg.timing_only:
        cfg.batch_size = 1
        cfg.epochs = 2

    #=================#
    # MAKE TRAINER
    #=================#

    #----------#
    # callback
    #----------#

    callback = mlutils.Callback(case_dir)

    if cfg.dataset in ['airfoil_dynamic', 'cylinder_flow']:
        callback = pdebench.TimeseriesCallback(case_dir, mesh=mesh)
    elif cfg.dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
        'shapenet_car', 'airfrans', 'am_small',
    ] or cfg.dataset.startswith('drivaerml'):
        callback = pdebench.RelL2Callback(case_dir, cfg.dataset, metadata['x_normalizer'], metadata['y_normalizer'])
    elif cfg.dataset in ['lpbf']:
        callback = am.FinaltimeCallback(case_dir, mesh=mesh, num_eval_cases=20)
    elif cfg.dataset in ['am_dynamic']:
        callback = am.TimeseriesCallback(case_dir, mesh=mesh, num_eval_cases=20, autoreg_start=1)

    # use scores callback in eval mode
    if cfg.model_type in [2,7] and cfg.evaluate and cfg.dataset in [
        'elasticity', 'darcy', 'airfoil_steady', 'shapenet_car', 'airfrans',
    ]:
        callback = pdebench.ScoresCallback(case_dir)

    #----------#
    # batch_size
    #----------#

    _batch_size  = cfg.batch_size
    if cfg.dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
    ]:
        batch_size_ = _batch_size_ = 5
    elif cfg.dataset in ['lpbf', 'airfoil_dynamic', 'cylinder_flow']:
        batch_size_ = _batch_size_ = 1
        assert _batch_size == WORLD_SIZE, f"Local batch size must be 1 for dataset {cfg.dataset}. Got batch_size={_batch_size} with WORLD_SIZE={WORLD_SIZE}."
    else:
        batch_size_ = _batch_size_ = 1

    gnn_loader = cfg.dataset in ['lpbf', 'airfoil_dynamic', 'cylinder_flow']

    #----------#
    # make_optimizer
    #----------#

    if cfg.optimizer == 'adamw':
        make_optimizer = pdebench.make_optimizer_adamw
    elif cfg.optimizer == 'lion':
        make_optimizer = pdebench.make_optimizer_lion
    else:
        raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Choose from adamw, lion.")

    #----------#
    # lossfun
    #----------#

    if cfg.dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
        'shapenet_car',
    ] or cfg.dataset.startswith('drivaerml'):
        if GLOBAL_RANK == 0:
            print(f"Using RelL2Loss for {cfg.dataset} dataset")
        lf = pdebench.RelL2Loss()
        def lossfun(yh, y):
            y_normalizer = metadata['y_normalizer'].to(y.device)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            return lf(yh, y)
    else:
        if GLOBAL_RANK == 0:
            print(f"Using MSELoss for {cfg.dataset} dataset")
        lossfun = torch.nn.MSELoss()

    #----------#
    # Trainer kwargs
    #----------#

    kw = dict(
        device=device, mixed_precision=cfg.mixed_precision, attn_backend=cfg.attn_backend, stats_every=cfg.epochs//10,
        # batch size
        _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
        # optimizer
        make_optimizer=make_optimizer, weight_decay=cfg.weight_decay, epochs=cfg.epochs,
        lossfun=lossfun, clip_grad_norm=cfg.clip_grad_norm,
        opt_betas=(cfg.opt_beta1, cfg.opt_beta2), opt_eps=cfg.opt_eps,
        # dataloader kwargs
        num_workers=cfg.num_workers, prefetch_factor=2, gnn_loader=gnn_loader,
    )

    #----------#
    # LR scheduler
    #----------#

    if cfg.schedule is None or cfg.schedule == 'ConstantLR':
        kw['lr'] = cfg.learning_rate
    elif cfg.schedule == 'OneCycleLR':
        kw['Schedule'] = 'OneCycleLR'
        kw['lr'] = cfg.learning_rate
        kw['one_cycle_pct_start'] = cfg.one_cycle_pct_start
        kw['one_cycle_div_factor'] = cfg.one_cycle_div_factor
        kw['one_cycle_final_div_factor'] = cfg.one_cycle_final_div_factor
        kw['one_cycle_three_phase'] = cfg.one_cycle_three_phase
    else:
        kw = dict(**kw, Schedule=cfg.schedule, lr=cfg.learning_rate,)

    #-------------#
    # make Trainer
    #-------------#

    trainer = mlutils.Trainer(model, _data, data_, **kw)
    trainer.add_callback('epoch_end', callback)

    #-------------#
    # batch_lossfun
    #-------------#
    if cfg.dataset in ['darcy']:

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
            return loss

        trainer.batch_lossfun = batch_lossfun

    elif cfg.dataset in ['lpbf']:

        lf = pdebench.RelL2Loss()
        def lossfun(yh, y):
            y_normalizer = metadata['y_normalizer'].to(y.device)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            return lf(yh, y)

        def batch_lossfun(trainer, model, batch):
            x  = batch.x.unsqueeze(0)
            y  = batch.y.unsqueeze(0)
            yh = model(x)
            return lossfun(yh, y)

        trainer.batch_lossfun = batch_lossfun

    elif cfg.dataset in ['airfoil_dynamic', 'cylinder_flow']:
        if GLOBAL_RANK == 0:
            print(f"Using masked loss for timeseries datasets {cfg.dataset}")
        batch_lossfun = am.MaskedLoss(mask=True)
        trainer.batch_lossfun = batch_lossfun

    #-------------#
    # load snapshot
    #-------------#

    if cfg.restart:
        callback.load(trainer)

    #=================#
    # TRAIN
    #=================#

    if cfg.train and cfg.epochs > 0:
        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    if cfg.evaluate:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer.make_dataloader()
        callback.load(trainer)
        callback(trainer, final=True)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Benchmarks transformer models on PDE datasets

    For training, run 

        python -m pdebench --train true ... <CONFIG>

    and the result will be saved to out/pdebench/<exp_name>/ckpt<01, 02, ...>.

    For evaluation, run

        python -m pdebench --evaluate true --exp_name <exp_name>

    and the model in the latest checkpoint out/pdebench/<exp_name>/ckptXX will be evaluated.

    For restarting from checkpoint, run

        python -m pdebench --restart true --exp_name <exp_name>

    and training will resume from the latest checkpoint in out/pdebench/<exp_name>/ckptXX.
    '''

    # case configuration
    train: bool = False
    evaluate: bool = False
    restart: bool = False

    exp_name: str = 'exp'
    seed: int = 0

    # dataset
    dataset: str = None
    num_workers: int = 0 # 0: auto, >0: manual

    # training arguments
    epochs: int = 100
    batch_size: int = 1
    weight_decay: float = 0e-0
    learning_rate: float = 1e-3
    schedule: str = 'OneCycleLR'
    one_cycle_pct_start:float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = False
    opt_beta1: float = 0.9
    opt_beta2: float = 0.999
    opt_eps: float = 1e-8
    clip_grad_norm: float = 1.0
    optimizer: str = 'adamw' # adamw, lion
    mixed_precision: bool = False
    attn_backend: str = None

    # timing run
    timing_only: bool = False

    # model
    model_type: int = 0 # 0: Transolver, 1: LNO, 2: FLARE, 3: Transformer, 4: GNOT, 5: UPT 6: PerceiverIO
    # Transolver
    conv2d: bool = False
    unified_pos: bool = False
    # PerceiverIO
    pcvr_cross_attn: bool = True
    ###
    # FLARE
    ###
    act: str = None
    channel_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    num_latents: int = 64
    #
    num_layers_kv_proj: int = 3
    num_layers_mlp: int = 3
    num_layers_in_out_proj: int = 2
    #
    mlp_ratio: float = 1.0
    kv_proj_ratio: float = 1.0
    in_out_proj_ratio: float = 1.0
    #
    out_proj_ln: bool = True
    # ablations
    shared_latents: bool = False
    num_latent_blocks: int = 0

#======================================================================#
if __name__ == "__main__":

    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if (cfg.train + cfg.evaluate + cfg.restart) != 1:
        msg = f"Invalid mode selection. Select one of train (got {cfg.train}), evaluate (got {cfg.evaluate}), restart (got {cfg.restart})."
        raise ValueError(msg)

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    if cfg.train:
        if os.path.exists(case_dir):
            # if exp_name already exists, append a number to make it unique
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.exp_name)])
            cfg.exp_name = cfg.exp_name + '_' + str(nd).zfill(2)
            case_dir = os.path.join(CASEDIR, cfg.exp_name)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if GLOBAL_RANK == 0:
            os.makedirs(case_dir)
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

    # load config from experiment directory
    if cfg.evaluate or cfg.restart:
        assert os.path.exists(case_dir), f"Experiment directory {case_dir} does not exist."
        config_file = os.path.join(case_dir, 'config.yaml')

        # save original config
        _cfg = cfg

        # load config from experiment directory
        if GLOBAL_RANK == 0:
            print(f'Loading config from {config_file}')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = Config(**{k: v for k, v in cfg.items() if k in Config.__annotations__})

        if _cfg.evaluate:
            cfg.evaluate = True
            cfg.train = False
        elif _cfg.restart:
            cfg.restart = True
            cfg.train = True

    if DISTRIBUTED:
        torch.distributed.barrier()

    main(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#