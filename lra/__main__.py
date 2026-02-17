#
import os
import yaml
import torch
from dataclasses import dataclass
from typing import Union, List, Optional
from jsonargparse import CLI

import lra
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
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')
    
#======================================================================#
def make_model(cfg, metadata, GLOBAL_RANK=0):

    vocab_size = metadata['vocab_size']
    num_labels = metadata['num_labels'] if not metadata['binary_classification'] else 1
    max_length = metadata['max_length']
    pool = cfg.pool
    pad_id = metadata.get('pad_id', None)

    assert cfg.pos_embed in ['sin', 'abs', 'rope'], f"Invalid pos_embed: {cfg.pos_embed}. Choose from: sin, abs, rope."

    if cfg.model_type == 'transformer':
        model_name = 'Transformer'
        backend_kwargs = dict(
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'flare':
        model_name = 'FLARE'
        kv_proj_hidden_dim = int(cfg.channel_dim * cfg.kv_proj_mlp_ratio)
        ffn_hidden_dim = int(cfg.channel_dim * cfg.ffn_mlp_ratio)

        assert cfg.attn_scale in ['sqrt', 'one'], f"Invalid attn_scale: {cfg.attn_scale}. Choose from: sqrt, one."
        assert cfg.channel_dim % cfg.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {cfg.channel_dim} and {cfg.num_heads}."

        head_dim = cfg.channel_dim // cfg.num_heads
        cfg.attn_scale = (head_dim ** -0.5) if cfg.attn_scale == 'sqrt' else 1.0

        backend_kwargs = dict(
            num_latents=cfg.num_latents,
            attn_scale=cfg.attn_scale,
            num_layers_kv_proj=cfg.num_layers_kv_proj,
            num_layers_ffn=cfg.num_layers_ffn,
            kv_proj_hidden_dim=kv_proj_hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'linformer':
        model_name = 'Linformer'
        backend_kwargs = dict(
            seq_len=max_length,
            k=cfg.linformer_k,
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'linear':
        model_name = 'LinearAttention'
        backend_kwargs = dict(
            kernel=cfg.kernel,
            norm_q=cfg.norm_q,
            norm_k=cfg.norm_k,
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'multilinear':
        model_name = 'MultilinearAttention'
        backend_kwargs = dict(
            num_states=cfg.num_states,
            num_layers_kv_proj=cfg.num_layers_kv_proj,
            kv_proj_mlp_ratio=cfg.kv_proj_mlp_ratio,
            kernel=cfg.kernel,
            norm_q=cfg.norm_q,
            norm_k=cfg.norm_k,
            qk_dim_ratio=cfg.qk_dim_ratio,
            num_layers_ffn=cfg.num_layers_ffn,
            ffn_mlp_ratio=cfg.ffn_mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'triple':
        model_name = 'TripleAttention'
        backend_kwargs = dict(
            num_layers_kv_proj=cfg.num_layers_kv_proj,
            kv_proj_mlp_ratio=cfg.kv_proj_mlp_ratio,
            kernel=cfg.kernel,
            norm_q=cfg.norm_q,
            norm_k=cfg.norm_k,
            qk_dim_ratio=cfg.qk_dim_ratio,
            use_triton=cfg.use_triton,
            num_layers_ffn=cfg.num_layers_ffn,
            ffn_mlp_ratio=cfg.ffn_mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'third_order':
        model_name = 'ThirdOrderAttention'
        backend_kwargs = dict(
            third_order_method=cfg.third_order_method,
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    elif cfg.model_type == 'performer':
        model_name = 'PerformerAttention'
        backend_kwargs = dict(
            nb_features=cfg.performer_nb_features,
            redraw_interval=cfg.performer_redraw_interval,
            normalize_inputs=cfg.performer_normalize_inputs,
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
    else:
        model_name = None

        backend_kwargs = dict(
            mlp_ratio=cfg.mlp_ratio,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        
    if model_name is None:
        model_name = cfg.model_type
        if GLOBAL_RANK == 0:
            print(f"No specific backend args for model type {cfg.model_type} found. Passing in default args.")

    if GLOBAL_RANK == 0:
        TRM = 'TRM' if cfg.trm else 'Standard'
        backend_kwargs_str = ''.join([f"\t{k}={v}\n" for k, v in backend_kwargs.items()])
        print(
            f"Using {TRM} {model_name}(task={cfg.task}) with\n"
            + f"\tchannel_dim={cfg.channel_dim}\n"
            + f"\tnum_blocks={cfg.num_blocks}\n"
            + f"\tnum_heads={cfg.num_heads}\n"
            + f"\tact={cfg.act}\n"
            + f"\trmsnorm={cfg.rmsnorm}\n"
            + f"\tpool={pool}\n"
            + f"\tpos_embed={cfg.pos_embed}\n"
            + backend_kwargs_str
        )
        
    if cfg.trm:
        model = lra.TRMWrapper(
            task=cfg.task,
            vocab_size=vocab_size,
            num_labels=num_labels,
            max_length=max_length,
            pool=pool,
            pad_id=pad_id,
            ###
            emb_drop=cfg.emb_drop,
            cls_drop=cfg.cls_drop,
            pos_embed=cfg.pos_embed,
            ###
            num_blocks=cfg.num_blocks,
            backend=cfg.model_type,
            channel_dim=cfg.channel_dim,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            ###
            trm_N_steps=cfg.trm_N_steps,
            trm_n=cfg.trm_n,
            trm_T=cfg.trm_T,
            ###
            **backend_kwargs,
        )
    else:
        model = lra.ModelWrapper(
            task=cfg.task,
            vocab_size=vocab_size,
            num_labels=num_labels,
            max_length=max_length,
            pool=pool,
            pad_id=pad_id,
            ###
            emb_drop=cfg.emb_drop,
            cls_drop=cfg.cls_drop,
            pos_embed=cfg.pos_embed,
            ###
            num_blocks=cfg.num_blocks,
            backend=cfg.model_type,
            channel_dim=cfg.channel_dim,
            num_heads=cfg.num_heads,
            act=cfg.act,
            rmsnorm=cfg.rmsnorm,
            **backend_kwargs,
        )
        
    return model

#======================================================================#
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0
    case_dir = os.path.join(CASEDIR, cfg.exp_name)
    
    #-------------#
    # Data
    #-------------#
    _data, data_, metadata = lra.load_dataset(cfg.task, DATADIR_BASE, GLOBAL_RANK=GLOBAL_RANK)

    if GLOBAL_RANK == 0:
        print(f"Loaded {metadata['task']} dataset: train {len(_data)}, val {len(data_)} samples with max length {metadata['max_length']}")

    #-------------#
    # Model
    #-------------#
    model = make_model(cfg, metadata, GLOBAL_RANK=GLOBAL_RANK)
    if GLOBAL_RANK == 0:
        # print(f"Model: {model}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    #-------------#
    # Callback
    #-------------#
    callback = lra.Callback(case_dir, metadata)

    #-------------#
    # Loss + stats
    #-------------#
    pad_id = metadata.get('pad_id', None)
    ignore_index = metadata.get('ignore_index', -100)
    if cfg.trm:
        batch_lossfun = lra.TRMBatchLossfun(N_steps=cfg.trm_N_steps, n=cfg.trm_n, T=cfg.trm_T)
    else:
        batch_lossfun = lra.SequenceClassificationLoss(
            binary_classification=metadata['binary_classification'],
            ignore_index=ignore_index
        )

    statsfun = lra.ClassificationStatsFun(
        binary_classification=metadata['binary_classification'],
        ignore_index=ignore_index
    )

    #-------------#
    # Optimizer
    #-------------#
    if cfg.optimizer in ['adamw', 'adam']:
        make_optimizer = lra.make_optimizer_adamw
    elif cfg.optimizer == 'lion':
        make_optimizer = lra.make_optimizer_lion
    else:
        raise NotImplementedError

    #-------------#
    # Collate function
    #-------------#
    if pad_id is not None:
        # Use padding collate for variable-length sequences (e.g., match3)
        _collate_fn = collate_fn_ = lra.PaddingCollate(pad_id=pad_id, ignore_index=ignore_index)
    else:
        # Use simple collate for fixed-length sequences
        _collate_fn = collate_fn_ = lra.simple_collate

    #-------------#
    # Trainer kwargs
    #-------------#
    kw = dict(
        # device & compilation
        device=device, mixed_precision=cfg.mixed_precision,
        compile_model=cfg.compile_model, static_graph=cfg.static_graph,
        ddp_find_unused_params=False, ddp_gradient_as_bucket_view=True,
        ema=cfg.ema, ema_decay=cfg.ema_decay,
        # batch size
        _batch_size=cfg.batch_size, batch_size_=cfg.batch_size, _batch_size_=cfg.batch_size,
        # optimizer
        make_optimizer=make_optimizer, weight_decay=cfg.weight_decay, epochs=cfg.epochs, steps=cfg.steps,
        batch_lossfun=batch_lossfun, clip_grad_norm=cfg.clip_grad_norm, grad_accumulation_steps=cfg.grad_accumulation_steps,
        opt_beta1=cfg.opt_beta1, opt_beta2=cfg.opt_beta2, opt_eps=cfg.opt_eps,
        # dataloader kwargs
        num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor,
        # statistics
        statsfun=statsfun,
        # collate function
        _collate_fn=_collate_fn, collate_fn_=collate_fn_,
        # # preprocess function
        # _preprocess_fn=_preprocess_fn, preprocess_fn_=preprocess_fn_,
    )

    if cfg.trm:
        kw['repeat_train_batch'] = cfg.trm_N_steps

    #-------------#
    # LR schedule
    #-------------#
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
        kw = dict(**kw, Schedule=cfg.schedule, lr=cfg.learning_rate)

    #-------------#
    # make trainer
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
    # Load snapshot
    #-------------#
    if cfg.restart:
        callback.load_latest_checkpoint(trainer)
    if cfg.load_weights_path is not None:
        trainer.load_weights(cfg.load_weights_path)

    #-------------#
    # Train
    #-------------#
    if cfg.train and (cfg.epochs > 0 or cfg.steps > 0):
        trainer.train()

    #-------------#
    # Evaluate
    #-------------#
    if cfg.evaluate:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer.make_dataloader()
        callback.load_latest_checkpoint(trainer)
        trainer.statistics()
        callback(trainer, final=True)

#======================================================================#
@dataclass
class Config:
    '''
    Benchmarks transformer models on language tasks

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

    # modes
    train: bool = False
    evaluate: bool = False
    restart: bool = False
    load_weights_path: str = None

    exp_name: str = 'exp'
    seed: int = 0

    # task
    task: str = 'listops' # lra tasks: listops, text, retrieval, image, pathfinder32, pathfinder128
    num_workers: int = 8
    prefetch_factor: int = 4

    # training
    epochs: int = 0
    steps: int = int(1e4)
    batch_size: int = 32
    optimizer: str = 'adamw'
    learning_rate: Union[float, List[float]] = 5e-4
    weight_decay: Union[float, List[float]] = 1e-4
    opt_beta1: Union[float, List[float]] = 0.9
    opt_beta2: Union[float, List[float]] = 0.98
    opt_eps: Union[float, List[float]] = 1e-8

    schedule: str = 'OneCycleLR'
    one_cycle_pct_start: float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e-2
    one_cycle_three_phase: bool = False
    one_cycle_cycle_momentum: bool = True
    one_cycle_base_momentum: float = 0.85
    one_cycle_max_momentum: float = 0.95
    one_cycle_anneal_strategy: str = 'cos'
    one_cycle_override_min_lr: float = None

    clip_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    mixed_precision: bool = True
    compile_model: bool = True
    static_graph: bool = True

    ema: bool = False
    ema_decay: float = 0.999

    # model
    model_type: str = 'transformer'

    # ModelWrapper
    num_blocks: int = 4 # 6, 8, 12
    channel_dim: int = 128 # 256, 512, 768
    num_heads: int = 8 # 8, 12
    act: str = None
    rmsnorm: bool = False
    pool: str = 'mean' # 'mean', 'max', 'cls'
    # Embedding
    emb_drop: float = 0.0
    cls_drop: float = 0.0
    pos_embed: str = 'abs' # 'sin', 'abs', 'rope'
    # Dropout
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    # Transformer
    mlp_ratio: float = 4.0
    # Linformer
    linformer_k: int = 256
    # Performer
    performer_nb_features: int = 256
    performer_redraw_interval: int = 0
    performer_normalize_inputs: bool = True
    # Linear
    kernel: str = 'identity' # elu, silu, silunorm, identity
    qk_dim_ratio: float = 1.0
    norm_q: bool = True
    norm_k: bool = True
    # Triple
    use_triton: bool = False
    # Multilinear
    num_states: int = 2
    # FLARE
    attn_scale: str = 'one' # 'one': 1.0, 'sqrt': 1/sqrt(D)
    num_latents: int = 128
    num_layers_kv_proj: int = 3
    num_layers_ffn: int = 3
    kv_proj_mlp_ratio: float = 1.0
    ffn_mlp_ratio: float = 1.0
    # ThirdOrderAttention
    third_order_method: str = 'third_order' # 'strassen', 'third_order'

    # TRM
    trm: bool = False
    trm_N_steps: int = 16
    trm_n: int = 6
    trm_T: int = 3

#======================================================================#
if __name__ == "__main__":
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    device = mlutils.select_device()

    cfg = CLI(Config, as_positional=False)

    if (cfg.train + cfg.evaluate + cfg.restart) != 1:
        msg = f"Invalid mode selection. Select one of train (got {cfg.train}), evaluate (got {cfg.evaluate}), restart (got {cfg.restart})."
        raise ValueError(msg)

    mlutils.set_seed(cfg.seed)
    mlutils.set_num_threads(cfg.num_workers)

    if cfg.train:
        cfg.exp_name = mlutils.get_next_exp_name(CASEDIR, cfg.exp_name)
        case_dir = os.path.join(CASEDIR, cfg.exp_name)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if GLOBAL_RANK == 0:
            os.makedirs(case_dir, exist_ok=True)
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

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

    mlutils.dist_finalize()
    exit()
#