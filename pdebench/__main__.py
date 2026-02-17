#
import torch

import os
import yaml
from jsonargparse import CLI
from typing import Union, List
from dataclasses import dataclass

# local
import pdebench
import mlutils

#======================================================================#
PROJDIR = mlutils.dotdot(os.path.dirname(__file__))
OUTNAME = os.path.basename(os.path.dirname(__file__))
CASEDIR = os.path.join(PROJDIR, 'out', OUTNAME)

mlutils.set_cache_path(mlutils.dotdot(PROJDIR))
os.makedirs(CASEDIR, exist_ok=True)

import socket
MACHINE = socket.gethostname()
if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti 11 GB
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')

#======================================================================#
def make_model(cfg, metadata, GLOBAL_RANK):
    """
    Create and configure model based on cfg.model_type.

    Args:
        cfg: Configuration object
        metadata: Dataset metadata from pdebench.load_dataset
        GLOBAL_RANK: Global rank for distributed training

    Returns:
        tuple: (updated cfg, model instance)
    """

    c_in = metadata['c_in']
    c_out = metadata['c_out']

    if cfg.model_type == 'transolver':
        #--------------------------------#
        # Transolver: https://arxiv.org/abs/2402.02366
        #--------------------------------#
        if cfg.use_defaults:
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
            cfg.one_cycle_override_min_lr = None
            cfg.clip_grad_norm = 0.1
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            # model params
            n_layers = 8
            n_hidden = 128 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 256
            slice_num = 64 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 32
            n_head = 8
            mlp_ratio = 1.0

        else:
            n_layers = cfg.num_blocks
            n_hidden = cfg.channel_dim
            slice_num = cfg.num_slices
            n_head = cfg.num_heads
            mlp_ratio = cfg.mlp_ratio

        if cfg.conv2d:
            model_name = 'Transolver_Structured_Mesh_2D'
            Model = pdebench.Transolver_Structured_Mesh_2D
            model_args = dict(
                space_dim=c_in, out_dim=c_out, fun_dim=0, n_hidden=n_hidden, n_layers=n_layers,
                n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
                H=metadata['H'], W=metadata['W'], unified_pos=cfg.unified_pos,
            )
        else:
            model_name = 'Transolver'
            Model = pdebench.Transolver
            model_args = dict(
                space_dim=c_in, out_dim=c_out, fun_dim=0,
                n_hidden=n_hidden, n_layers=n_layers,
                n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
            )

    elif cfg.model_type == 'transolver++':
        #--------------------------------#
        # Transolver++
        #--------------------------------#
        if cfg.use_defaults:
            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500

            if cfg.dataset in ['elasticity',]:
                cfg.batch_size = 8
            elif cfg.dataset in ['darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 4
            elif cfg.dataset.startswith('drivaerml') or cfg.dataset in ['lpbf']:
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
            cfg.one_cycle_override_min_lr = None
            cfg.clip_grad_norm = 0.1
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5
            
            # model params
            n_layers = 8
            n_hidden = 128
            slice_num = 64
            n_head = 8
            mlp_ratio = 1.0
        else:
            n_layers = cfg.num_blocks
            n_hidden = cfg.channel_dim
            slice_num = cfg.num_slices
            n_head = cfg.num_heads
            mlp_ratio = cfg.mlp_ratio

        model_name = 'TransolverPlusPlus'
        Model = pdebench.TransolverPlusPlus
        model_args = dict(
            space_dim=c_in, out_dim=c_out, fun_dim=0, n_hidden=n_hidden, n_layers=n_layers,
            n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
        )

    elif cfg.model_type == 'lno':
        #--------------------------------#
        # LNO: https://github.com/L-I-M-I-T/LatentNeuralOperator
        #--------------------------------#
        if cfg.use_defaults:
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
            cfg.one_cycle_override_min_lr = None
            cfg.clip_grad_norm = 1000.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 5e-5

            # model params
            n_head = 8
            n_mode = 256
            n_dim = 192 if cfg.dataset in ['elasticity'] else 128
            n_layer = 3 if cfg.dataset in ['elasticity'] else 2
            n_block = 8 if cfg.dataset in ['pipe', 'airfoil_steady'] else 4
            
        else:
            n_head = cfg.num_heads
            n_mode = cfg.num_modes
            n_dim = cfg.channel_dim
            n_layer = cfg.num_layers_kv_proj
            n_block = cfg.num_blocks

        model_name = 'LNO'
        Model = pdebench.LNO
        model_args = dict(
            n_block=n_block, n_mode=n_mode, n_dim=n_dim, n_head=n_head, n_layer=n_layer, act="GELU",
            x_dim=c_in, y1_dim=c_in, y2_dim=c_out, model_attr={"time": metadata['time_cond'],}
        )

    elif cfg.model_type == 'gnot':
        #--------------------------------#
        # GNOT
        #--------------------------------#
        if cfg.use_defaults:
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
            cfg.one_cycle_override_min_lr = None
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

            # model params
            n_layers = 8
            n_hidden = 128
            mlp_ratio = 2.0
            n_experts = 3
            n_head = 8
        else:
            n_layers = cfg.num_blocks
            n_hidden = cfg.channel_dim
            mlp_ratio = cfg.mlp_ratio
            n_experts = cfg.num_experts
            n_head = cfg.num_heads

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

        model_name = 'GNOT'
        Model = pdebench.GNOT
        model_args = dict(
            n_experts=n_experts, n_heads=n_head, n_hidden=n_hidden,
            n_layers=n_layers, mlp_ratio=mlp_ratio, unified_pos=unified_pos,
            geotype=geotype, shapelist=shapelist, ref=ref, space_dim=c_in, fun_dim=0, out_dim=c_out,
        )

    elif cfg.model_type == 'upt':
        #--------------------------------#
        # UPT (Universal Physics Transformer)
        #--------------------------------#
        raise NotImplementedError("UPT is not implemented yet.")

    elif cfg.model_type == 'lamo':
        #--------------------------------#
        # LaMO
        #--------------------------------#
        if cfg.use_defaults:
            n_layers = 8
            n_hidden = 128 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 256
            slice_num = 64 if cfg.dataset not in ['airfrans', 'shapenet_car', 'navier_stokes'] else 32
            n_head = 8
            mlp_ratio = 1.0
        else:
            n_layers = cfg.num_blocks
            n_hidden = cfg.channel_dim
            slice_num = cfg.num_slices
            n_head = cfg.num_heads
            mlp_ratio = cfg.mlp_ratio

        if cfg.conv2d:
            model_name = 'LaMO_Structured_Mesh_2D'
            Model = pdebench.LaMO_Structured_Mesh_2D
            model_args = dict(
                space_dim=c_in, out_dim=c_out, fun_dim=0,
                n_hidden=n_hidden, n_layers=n_layers,
                n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
                H=metadata['H'], W=metadata['W'],
                unified_pos=cfg.unified_pos,
            )
        else:
            model_name = 'LaMO'
            Model = pdebench.LaMO
            model_args = dict(
                space_dim=c_in, out_dim=c_out, fun_dim=0,
                n_hidden=n_hidden, n_layers=n_layers,
                n_head=n_head, mlp_ratio=mlp_ratio, slice_num=slice_num,
            )

    elif cfg.model_type == 'perceiverio':
        #--------------------------------#
        # PerceiverIO
        #--------------------------------#
        if cfg.use_defaults:
            cfg.epochs = 250 if cfg.dataset in ['shapenet_car', 'lpbf'] else 500
            if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
                cfg.batch_size = 2
            elif cfg.dataset in ['shapenet_car', 'lpbf'] or cfg.dataset.startswith('drivaerml'):
                cfg.batch_size = 1
            else:
                raise ValueError(f"Batch size not set for dataset {cfg.dataset}")
            cfg.learning_rate = 1e-3
            cfg.opt_beta1 = 0.9
            cfg.opt_beta2 = 0.999
            cfg.schedule = 'OneCycleLR'
            cfg.one_cycle_pct_start = 0.1
            cfg.one_cycle_div_factor = 25
            cfg.one_cycle_final_div_factor = 1e4
            cfg.one_cycle_override_min_lr = None
            cfg.clip_grad_norm = 1.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            # model params
            channel_dim = 128
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None
            num_latents = 512
            cross_attn = cfg.pcvr_cross_attn
        else:
            channel_dim = cfg.channel_dim
            num_blocks = cfg.num_blocks
            num_heads = cfg.num_heads
            mlp_ratio = cfg.mlp_ratio
            act = cfg.act
            num_latents = cfg.num_latents
            cross_attn = cfg.pcvr_cross_attn

        model_name = 'PerceiverIO'
        Model = pdebench.PerceiverIO
        model_args = dict(
            in_dim=c_in, out_dim=c_out, channel_dim=channel_dim,
            num_blocks=num_blocks, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_latents=num_latents, act=act,
            cross_attn=cross_attn,
        )

    elif cfg.model_type == 'transformer':
        #--------------------------------#
        # Vanilla Transformer (softmax attention)
        #--------------------------------#
        if cfg.use_defaults:
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
            cfg.one_cycle_override_min_lr = None
            cfg.clip_grad_norm = 1.0
            if cfg.dataset in ['shapenet_car']:
                cfg.weight_decay = 5e-2
            elif cfg.dataset in ['drivaerml_40k']:
                cfg.weight_decay = 1e-4
            elif cfg.dataset in ['lpbf']:
                cfg.weight_decay = 1e-4
            else:
                cfg.weight_decay = 1e-5

            # model params
            channel_dim = 80
            num_blocks = 8
            num_heads = channel_dim // 16
            mlp_ratio = 4.0
            act = None
            rmsnorm = False
            out_proj_norm = True
            num_layers_in_out_proj = 2

        else:
            channel_dim = cfg.channel_dim
            num_blocks = cfg.num_blocks
            num_heads = cfg.num_heads
            mlp_ratio = cfg.mlp_ratio
            act = cfg.act
            rmsnorm = cfg.rmsnorm
            out_proj_norm = cfg.out_proj_norm
            num_layers_in_out_proj = cfg.num_layers_in_out_proj

        backend_kwargs = dict(
            mlp_ratio=mlp_ratio,
        )

        model_name = 'Transformer'
        Model = pdebench.TransformerWrapper
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=channel_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            act=act,
            rmsnorm=rmsnorm,
            #
            out_proj_norm=out_proj_norm,
            num_layers_in_out_proj=num_layers_in_out_proj,
            #
            backend='transformer',
            **backend_kwargs,
        )

    elif cfg.model_type == 'linformer':
        #--------------------------------#
        # Linformer
        #--------------------------------#
        backend_kwargs = dict(
            mlp_ratio=cfg.mlp_ratio,
            seq_len=metadata['max_length'],
            k=cfg.linformer_k,
        )
        
        model_name = 'Linformer'
        Model = pdebench.TransformerWrapper
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            #
            out_proj_norm=cfg.out_proj_norm,
            num_layers_in_out_proj=cfg.num_layers_in_out_proj,
            #
            backend='linformer',
            **backend_kwargs,
        )

    elif cfg.model_type == 'linear':
        #--------------------------------#
        # Linear attention
        #--------------------------------#
        backend_kwargs = dict(
            mlp_ratio=cfg.mlp_ratio,
            kernel=cfg.kernel,
            norm_q=cfg.norm_q,
            norm_k=cfg.norm_k,
        )

        model_name = 'Linear'
        Model = pdebench.TransformerWrapper
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            #
            out_proj_norm=cfg.out_proj_norm,
            num_layers_in_out_proj=cfg.num_layers_in_out_proj,
            #
            backend='linear',
            **backend_kwargs,
        )

    # elif cfg.model_type in ['multilinear', 'triple', 'quad', 'strassen']:
    #     #--------------------------------#
    #     # Multilinear, Triple, Quad, Strassen attention
    #     #--------------------------------#
    #     if GLOBAL_RANK == 0:
    #         name_dict = {
    #                 'multilinear': 'MultilinearAttention',
    #                 'triple': 'TripleAttention',
    #                 'quad': 'QuadAttention',
    #                 'strassen': 'StrassenAttention',
    #                 }
    #     backend_kwargs = dict(
    #         kernel=cfg.kernel,
    #         norm_q=cfg.norm_q,
    #         norm_k=cfg.norm_k,
    #         num_layers_kv_proj=cfg.num_layers_kv_proj,
    #         kv_proj_mlp_ratio=cfg.kv_proj_mlp_ratio,
    #         qk_dim_ratio=cfg.qk_dim_ratio,
    #         #
    #         num_layers_ffn=cfg.num_layers_ffn,
    #         ffn_mlp_ratio=cfg.ffn_mlp_ratio,
    #     )

    #     if cfg.model_type == 'multilinear':
    #         backend_kwargs['num_states'] = cfg.num_states
    #     elif cfg.model_type == 'triple':
    #         backend_kwargs['use_triton'] = cfg.use_triton
            
    #     model_name = name_dict[cfg.model_type]
    #     Model = pdebench.TransformerWrapper
    #     model_args = dict(
    #         in_dim=c_in,
    #         out_dim=c_out,
    #         channel_dim=cfg.channel_dim,
    #         num_blocks=cfg.num_blocks,
    #         num_heads=cfg.num_heads,
    #         act=cfg.act,
    #         rmsnorm=cfg.rmsnorm,
    #         #
    #         out_proj_norm=cfg.out_proj_norm,
    #         num_layers_in_out_proj=cfg.num_layers_in_out_proj,
    #         #
    #         backend=cfg.model_type,
    #         **backend_kwargs,
    #     )
    # elif cfg.model_type == 'triple1':
    #     #--------------------------------#
    #     # Triple1 attention
    #     #--------------------------------#
    #     backend_kwargs = dict(
    #         mlp_ratio=cfg.mlp_ratio,
    #         qk_dim_ratio=cfg.qk_dim_ratio,
    #         use_triton=cfg.use_triton,
    #     )

    #     model_name = 'Triple1Attention'
    #     Model = pdebench.TransformerWrapper
    #     model_args = dict(
    #         in_dim=c_in,
    #         out_dim=c_out,
    #         channel_dim=cfg.channel_dim,
    #         num_blocks=cfg.num_blocks,
    #         num_heads=cfg.num_heads,
    #         act=cfg.act,
    #         rmsnorm=cfg.rmsnorm,
    #         #
    #         out_proj_norm=cfg.out_proj_norm,
    #         num_layers_in_out_proj=cfg.num_layers_in_out_proj,
    #         #
    #         backend=cfg.model_type,
    #         **backend_kwargs,
    #     )

    elif cfg.model_type == 'flare':
        #--------------------------------#
        # FLARE
        #--------------------------------#
        assert cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {cfg.attn_scale}. Choose from: sqrt, one."
        assert cfg.channel_dim % cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {cfg.channel_dim} and {cfg.num_heads}."
        head_dim = cfg.channel_dim // cfg.num_heads
        cfg.attn_scale = (head_dim ** -0.5) if cfg.attn_scale == 'sqrt' else 1.0

        backend_kwargs = dict(
            attn_scale=cfg.attn_scale,
            num_latents=cfg.num_latents,
            num_layers_k_proj=cfg.num_layers_k_proj,
            num_layers_v_proj=cfg.num_layers_v_proj,
            k_proj_mlp_ratio=cfg.k_proj_mlp_ratio,
            v_proj_mlp_ratio=cfg.v_proj_mlp_ratio,
            num_layers_ffn=cfg.num_layers_ffn,
            ffn_mlp_ratio=cfg.ffn_mlp_ratio,
            qk_norm=cfg.qk_norm,
        )

        model_name = 'FLARE'
        Model = pdebench.FLAREModel
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            out_proj_norm=cfg.out_proj_norm,
            num_layers_in_out_proj=cfg.num_layers_in_out_proj,
            **backend_kwargs,
        )

    elif cfg.model_type == 'flare_experimental':
        #--------------------------------#
        # FLARE
        #--------------------------------#
        assert cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {cfg.attn_scale}. Choose from: sqrt, one."
        assert cfg.channel_dim % cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {cfg.channel_dim} and {cfg.num_heads}."
        head_dim = cfg.channel_dim // cfg.num_heads
        cfg.attn_scale = (head_dim ** -0.5) if cfg.attn_scale == 'sqrt' else 1.0

        backend_kwargs = dict(
            attn_scale=cfg.attn_scale,
            num_latents=cfg.num_latents,
            num_layers_k_proj=cfg.num_layers_k_proj,
            num_layers_v_proj=cfg.num_layers_v_proj,
            k_proj_mlp_ratio=cfg.k_proj_mlp_ratio,
            v_proj_mlp_ratio=cfg.v_proj_mlp_ratio,
            num_layers_ffn=cfg.num_layers_ffn,
            ffn_mlp_ratio=cfg.ffn_mlp_ratio,
            qk_norm=cfg.qk_norm,
        )

        model_name = 'FLARE-Experimental'
        Model = pdebench.FLAREExperimentalModel
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            out_proj_norm=cfg.out_proj_norm,
            num_layers_in_out_proj=cfg.num_layers_in_out_proj,
            **backend_kwargs,
        )

    # elif cfg.model_type == 'loopy':
    #     #--------------------------------#
    #     # Loopy Transformer
    #     #--------------------------------#
    #     assert cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {cfg.attn_scale}. Choose from: sqrt, one."
    #     assert cfg.channel_dim % cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {cfg.channel_dim} and {cfg.num_heads}."
    #     head_dim = cfg.channel_dim // cfg.num_heads
    #     cfg.attn_scale = (head_dim ** -0.5) if cfg.attn_scale == 'sqrt' else 1.0

    #     backend_kwargs = dict(
    #         attn_scale=cfg.attn_scale,
    #         num_latents=cfg.num_latents,
    #         num_layers_kv_proj=cfg.num_layers_kv_proj,
    #         kv_proj_mlp_ratio=cfg.kv_proj_mlp_ratio,
    #         num_layers_ffn=cfg.num_layers_ffn,
    #         ffn_mlp_ratio=cfg.ffn_mlp_ratio,
    #     )
        
    #     model_name = 'Loopy'
    #     Model = pdebench.LoopyWrapper
    #     model_args = dict(
    #         in_dim=c_in,
    #         out_dim=c_out,
    #         channel_dim=cfg.channel_dim,
    #         num_blocks=cfg.num_blocks,
    #         num_heads=cfg.num_heads,
    #         act=cfg.act,
    #         rmsnorm=cfg.rmsnorm,
    #         out_proj_norm=cfg.out_proj_norm,
    #         num_layers_in_out_proj=cfg.num_layers_in_out_proj,
    #         num_passes=cfg.num_passes,
    #         **backend_kwargs,
    #     )

    # elif cfg.model_type == 'unloopy':
    #     #--------------------------------#
    #     # Unloopy Transformer
    #     #--------------------------------#
    #     assert cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {cfg.attn_scale}. Choose from: sqrt, one."
    #     assert cfg.channel_dim % cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {cfg.channel_dim} and {cfg.num_heads}."
    #     head_dim = cfg.channel_dim // cfg.num_heads
    #     cfg.attn_scale = (head_dim ** -0.5) if cfg.attn_scale == 'sqrt' else 1.0

    #     backend_kwargs = dict(
    #         attn_scale=cfg.attn_scale,
    #         num_latents=cfg.num_latents,
    #         num_layers_kv_proj=cfg.num_layers_kv_proj,
    #         kv_proj_mlp_ratio=cfg.kv_proj_mlp_ratio,
    #         num_layers_ffn=cfg.num_layers_ffn,
    #         ffn_mlp_ratio=cfg.ffn_mlp_ratio,
    #     )
        
    #     model_name = 'Unloopy'
    #     Model = pdebench.UnloopyWrapper
    #     model_args = dict(
    #         in_dim=c_in,
    #         out_dim=c_out,
    #         channel_dim=cfg.channel_dim,
    #         num_blocks=cfg.num_blocks,
    #         num_heads=cfg.num_heads,
    #         act=cfg.act,
    #         rmsnorm=cfg.rmsnorm,
    #         out_proj_norm=cfg.out_proj_norm,
    #         num_layers_in_out_proj=cfg.num_layers_in_out_proj,
    #         shared_ffn=cfg.shared_ffn,
    #         shared_att=cfg.shared_att,
    #         gating=cfg.gating,
    #         num_layers_gating_proj=cfg.num_layers_gating_proj,
    #         gating_proj_mlp_ratio=cfg.gating_proj_mlp_ratio,
    #         **backend_kwargs,
    #     )

    elif cfg.model_type == 'flare_ablations':
        #--------------------------------#
        # BigFLARE (ablations)
        #--------------------------------#
        model_name = 'BigFLARE'
        Model = pdebench.BigFLAREModel
        model_args = dict(
            in_dim=c_in,
            out_dim=c_out,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_latents=cfg.num_latents,
            num_heads=cfg.num_heads,
            act=cfg.act,
            num_layers_kv_proj=cfg.num_layers_kv_proj,
            num_layers_mlp=cfg.num_layers_ffn,
            num_layers_in_out_proj=cfg.num_layers_in_out_proj,
            mlp_ratio=cfg.ffn_mlp_ratio,
            kv_proj_ratio=cfg.kv_proj_mlp_ratio,
            in_out_proj_ratio=cfg.in_out_proj_ratio,
            out_proj_ln=cfg.out_proj_norm,
            shared_latents=cfg.shared_att,
            num_latent_blocks=cfg.num_passes,
        )

    else:
        #--------------------------------#
        # No model selected
        #--------------------------------#
        raise NotImplementedError(f"Model type {cfg.model_type} not implemented.")

    if GLOBAL_RANK == 0:
        model_args_str = ''.join([f"\t{k}={v}\n" for k, v in model_args.items()])
        print(f"Using {model_name}(c_in={c_in}, c_out={c_out}) with\n" + model_args_str)

    model = Model(**model_args)

    return cfg, model

#======================================================================#
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    mesh = cfg.model_type in ['mesh_model',]  # placeholder for mesh-based models
    _data, data_, metadata = pdebench.load_dataset(cfg.dataset, DATADIR_BASE, PROJDIR, mesh=mesh)

    if metadata is None:
        raise ValueError("metadata is None. Check pdebench.load_dataset and your dataset path/configuration.")

    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")
        # print(f"Number of points: {len(next(_data))}")

    #=================#
    # MODEL
    #=================#

    cfg, model = make_model(cfg, metadata, GLOBAL_RANK)

    # Handle time-conditioned models
    if metadata['time_cond']:
        raise NotImplementedError("Time-conditioned models not implemented in this repository.")

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
        from pdebench.callbacks_timeseries import TimeseriesCallback
        callback = TimeseriesCallback(case_dir, mesh=mesh)
    elif cfg.dataset in [
        'elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
        'shapenet_car', 'airfrans', 'am_small',
    ] or cfg.dataset.startswith('drivaerml'):
        callback = pdebench.RelL2Callback(case_dir, cfg.dataset, metadata['x_normalizer'], metadata['y_normalizer'])
    elif cfg.dataset in ['lpbf']:
        import am
        callback = am.FinaltimeCallback(case_dir, mesh=mesh, num_eval_cases=20)
    elif cfg.dataset in ['am_dynamic']:
        import am

        callback = am.TimeseriesCallback(case_dir, mesh=mesh, num_eval_cases=20, autoreg_start=1)

    # use scores callback in eval mode
    if cfg.model_type in ['flare', 'flare_ablations'] and cfg.evaluate and cfg.dataset in [
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

    if cfg.optimizer in ['adamw', 'adam']:
        make_optimizer = pdebench.make_optimizer_adamw
    elif cfg.optimizer == 'lion':
        make_optimizer = pdebench.make_optimizer_lion
    elif cfg.optimizer == 'muon':
        cfg.one_cycle_cycle_momentum = False
        make_optimizer = pdebench.make_optimizer_muon
    else:
        raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Choose from adamw, lion, muon.")

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
        # device & compilation
        device=device, mixed_precision=cfg.mixed_precision,
        compile_model=cfg.compile_model, static_graph=cfg.static_graph,
        ddp_find_unused_params=False, ddp_gradient_as_bucket_view=True,
        ema=cfg.ema, ema_decay=cfg.ema_decay,
        # batch size
        _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
        # optimizer
        make_optimizer=make_optimizer, weight_decay=cfg.weight_decay, epochs=cfg.epochs, steps=cfg.steps,
        lossfun=lossfun, clip_grad_norm=cfg.clip_grad_norm,
        opt_beta1=cfg.opt_beta1, opt_beta2=cfg.opt_beta2, opt_eps=cfg.opt_eps,
        # dataloader kwargs
        num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor, gnn_loader=gnn_loader,
    )

    #----------#
    # LR scheduler
    #----------#

    if cfg.schedule is None or cfg.schedule == 'ConstantLR':
        kw['lr'] = cfg.learning_rate
    elif cfg.schedule == 'OneCycleLR':

        if cfg.one_cycle_override_min_lr is not None:
            cfg.one_cycle_div_factor = cfg.learning_rate / cfg.one_cycle_override_min_lr
            cfg.one_cycle_final_div_factor = 1.0

        kw['Schedule'] = 'OneCycleLR'
        kw['lr'] = cfg.learning_rate
        kw['one_cycle_pct_start'] = cfg.one_cycle_pct_start
        kw['one_cycle_div_factor'] = cfg.one_cycle_div_factor
        kw['one_cycle_final_div_factor'] = cfg.one_cycle_final_div_factor
        kw['one_cycle_three_phase'] = cfg.one_cycle_three_phase
        kw['one_cycle_cycle_momentum'] = cfg.one_cycle_cycle_momentum
        kw['one_cycle_base_momentum'] = cfg.one_cycle_base_momentum
        kw['one_cycle_max_momentum'] = cfg.one_cycle_max_momentum
        kw['one_cycle_anneal_strategy'] = cfg.one_cycle_anneal_strategy
    else:
        kw = dict(**kw, Schedule=cfg.schedule, lr=cfg.learning_rate,)

    #-------------#
    # make Trainer
    #-------------#

    trainer = mlutils.Trainer(model, _data, data_, **kw)

    #-------------#
    # add callback
    #-------------#
    if trainer.train_based_on_epochs:
        trainer.add_callback('epoch_end', callback)
    else:
        trainer.add_callback('batch_end', callback)

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
            # loss = l2
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

        import am

        if GLOBAL_RANK == 0:
            print(f"Using masked loss for timeseries datasets {cfg.dataset}")
        batch_lossfun = am.MaskedLoss(mask=True)
        trainer.batch_lossfun = batch_lossfun

    #-------------#
    # load snapshot
    #-------------#
    
    if cfg.restart:
        callback.load_latest_checkpoint(trainer)
    if cfg.load_weights_path is not None:
        trainer.load_weights(cfg.load_weights_path)
    
    #=================#
    # TRAIN
    #=================#

    if cfg.train and (cfg.epochs > 0 or cfg.steps > 0):
        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    if cfg.evaluate:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer.make_dataloader()
        callback.load_latest_checkpoint(trainer)
        trainer.statistics()
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

    For loading weights, run

        python -m pdebench --load_weights_path <path_to_weights> ... <CONFIG>

    and the weights will be loaded from the specified path.
    '''

    # case configuration
    train: bool = False
    evaluate: bool = False
    restart: bool = False
    load_weights_path: str = None

    exp_name: str = 'exp'
    seed: int = 0

    # dataset
    dataset: str = None
    num_workers: int = 0
    prefetch_factor: int = None

    # training arguments
    epochs: int = 100
    steps: int = 0
    batch_size: int = 1
    # Optimizer
    optimizer: str = 'adamw' # adamw, lion, muon
    learning_rate: Union[float, List[float]] = 1e-3
    weight_decay: Union[float, List[float]] = 0e-0
    opt_beta1: Union[float, List[float]] = 0.9
    opt_beta2: Union[float, List[float]] = 0.999
    opt_eps: Union[float, List[float]] = 1e-8
    # Scheduler
    schedule: str = 'OneCycleLR'
    # OneCycleLR
    one_cycle_pct_start:float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = False
    one_cycle_cycle_momentum: bool = True
    one_cycle_base_momentum: float = 0.85
    one_cycle_max_momentum: float = 0.95
    one_cycle_anneal_strategy: str = 'cos'
    one_cycle_override_min_lr: float = None

    clip_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    mixed_precision: bool = False
    compile_model: bool = True
    static_graph: bool = True

    ema: bool = True
    ema_decay: float = 0.999

    # timing run
    timing_only: bool = False

    # model
    model_type: str = 'flare' # transolver(++), lno, transformer, gnot, perceiverio, flare, flare_ablations, lamo
    use_defaults: bool = False # use hardcoded default hyperparameters

    # Used in all models
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    act: str = None
    rmsnorm: bool = False
    # Transformer
    mlp_ratio: float = 4.0
    # Linformer
    linformer_k: int = 256
    # Linear
    kernel: str = 'identity' # elu, silu, silunorm, identity
    qk_dim_ratio: float = 1.0
    norm_q: bool = True
    norm_k: bool = True
    # Triple
    use_triton: bool = False
    # Multilinear
    num_states: int = 2
    num_layers_kv_proj: int = 3
    kv_proj_mlp_ratio: float = 1.0
    # FLARE
    attn_scale: str = 'one' # 'one': 1.0, 'sqrt': 1/sqrt(D)
    num_latents: int = 64
    num_layers_k_proj: int = 3
    num_layers_v_proj: int = 3
    k_proj_mlp_ratio: float = 1.0
    v_proj_mlp_ratio: float = 1.0
    num_layers_ffn: int = 3
    ffn_mlp_ratio: float = 1.0
    qk_norm: bool = False
    # Loopy
    num_passes: int = 1
    # Unloopy
    shared_ffn: bool = False
    shared_att: bool = False
    gating: bool = False
    num_layers_gating_proj: int = 3
    gating_proj_mlp_ratio: float = 1.0
    # Input/output projection
    num_layers_in_out_proj: int = 2
    in_out_proj_ratio: float = 1.0
    out_proj_norm: bool = True
    # Transolver
    conv2d: bool = False
    unified_pos: bool = False
    num_slices: int = 64
    # LNO
    num_modes: int = 256
    # GNOT
    num_experts: int = 3
    # Perceiver
    pcvr_cross_attn: bool = False

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

    if cfg.train:
        cfg.exp_name = mlutils.get_next_exp_name(CASEDIR, cfg.exp_name)
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
        case_dir = os.path.join(CASEDIR, cfg.exp_name)
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
