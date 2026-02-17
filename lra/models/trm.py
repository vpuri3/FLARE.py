#
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'TRMWrapper',
    'TRMBatchLossfun',
]

import mlutils
from .backends import MODEL_TYPES
from .embeddings import TokenEmb, PosEmb, RotaryPositionalEmbeddings

#======================================================================#
class TRMWrapper(nn.Module):
    def __init__(self,
        # Task related
        task: str,
        vocab_size: int,
        num_labels: int,
        max_length: int = 256,
        pool: str = 'mean',
        pad_id: int = None,
        ###
        emb_drop: float = 0.0,
        cls_drop: float = 0.0,
        pos_embed: str = 'sin', # 'sin', 'abs', 'rope'
        ###
        num_blocks: int = 2,
        backend: str = 'transformer',
        channel_dim: int = 128,
        num_heads: int = 4,
        act: str = None,
        rmsnorm: bool = False,
        ###
        trm_N_steps: int = 10,
        trm_n: int = 6,
        trm_T: int = 3,
        ###
        **backend_kwargs,
    ):
        super().__init__()
        self.task = task
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.pos_embed_type = pos_embed

        self.trm_N_steps = trm_N_steps
        self.trm_n = trm_n
        self.trm_T = trm_T

        #--------------------------------#
        # Pooling strategy
        #--------------------------------#
        if self.task in [
                'sudoku', 'match2', 'match3',
                'binary_relation_composition', 'quotient_binary_relation_composition'
            ]:
            self.pool = None
        else:
            assert pool in ['mean', 'max'], f"Invalid pool: {pool}. Acceptable values are 'mean' or 'max'. 'cls' is not supported."
            self.pool = pool

        #--------------------------------#
        # Token embeddings
        #--------------------------------#
        # If pad_id is provided, increase vocab_size by 1 to accommodate padding token
        # Embedding will use vocab_size+1 embeddings, with pad_id as the padding_idx
        embedding_vocab_size = vocab_size + 1 if pad_id is not None else vocab_size
        self.token_emb = TokenEmb(
            vocab_size=embedding_vocab_size,
            channel_dim=channel_dim,
            drop=emb_drop,
            padding_idx=pad_id,
        )

        #--------------------------------#
        # Positional embeddings (only for abs/sin - rope is handled separately)
        #--------------------------------#
        if pos_embed in ['abs', 'sin']:
            self.pos_emb = PosEmb(
                channel_dim=channel_dim,
                max_length=max_length,
                pos_embed=pos_embed,
            )
        else:
            self.pos_emb = None

        if pos_embed == 'rope':
            self.rope = RotaryPositionalEmbeddings(channel_dim // num_heads, max_length)
        else:
            self.rope = None

        #--------------------------------#
        # y, z, init
        #--------------------------------#
        self.y_init = nn.Parameter(torch.empty(channel_dim))
        self.z_init = nn.Parameter(torch.empty(channel_dim))
        nn.init.trunc_normal_(self.y_init, std=0.02)
        nn.init.trunc_normal_(self.z_init, std=0.02)

        #--------------------------------#
        # Backend
        #--------------------------------#
        Block = MODEL_TYPES.get(backend, None)
        if Block is None:
            raise NotImplementedError(f"Backend {backend} not implemented. Available backends:\n{list(MODEL_TYPES.keys())}.")

        self.blocks = nn.ModuleList([Block(
            channel_dim=channel_dim,
            num_heads=num_heads,
            act=act,
            rmsnorm=rmsnorm,
            rope=self.rope,
            **backend_kwargs,
        ) for _ in range(num_blocks)])

        #--------------------------------#
        # Classifier
        #--------------------------------#
        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.final_norm = Norm(channel_dim)

        self.cls_proj = nn.Sequential(
            nn.Dropout(cls_drop),
            Norm(channel_dim),
            nn.Linear(channel_dim, num_labels),
        )

        self.halt_proj = nn.Sequential(
            nn.Dropout(cls_drop),
            Norm(channel_dim),
            nn.Linear(channel_dim, 1),
        )

        #--------------------------------#
        # Weight initialization
        #--------------------------------#
        self.init_weights()

    def init_weights(self):
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)

    def apply_blocks(self, x: torch.Tensor, add_pos: bool = True):
        B, N, C = x.shape
        device = x.device

        pos = 0
        if add_pos and (self.pos_emb is not None): 
            pos = self.pos_emb(B, N, device)
            pos = pos.unsqueeze(0).expand(B, -1, -1) if pos.dim() == 2 else pos # [B, N, C]

        for block in self.blocks:
            x = block(x + pos)

        x = self.final_norm(x)

        return x

    def maxpool(self, x: torch.Tensor):
        return x.max(dim=1).values

    def meanpool(self, x: torch.Tensor):
        return x.mean(dim=1)

    def apply_pool(self, x: torch.Tensor):
        if self.pool == 'mean':
            return self.meanpool(x)
        elif self.pool == 'max':
            return self.maxpool(x)
        return x

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int = 6):
        for _ in range(n):
            z = self.apply_blocks(x + y + z)
        y = self.apply_blocks(y + z)
        return y, z

    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int = 6, T: int = 3):
        with torch.no_grad():
            for _ in range(T-1):
                y, z = self.latent_recursion(x, y, z, n)

        y, z = self.latent_recursion(x, y, z, n)
        cls_logits = self.cls_proj(self.apply_pool(y)) # [B, N, V]
        halt_logits = self.halt_proj(self.maxpool(y))  # [B, 1]

        return (y.detach(), z.detach()), cls_logits, halt_logits

    def init_y_z(self, input_ids: torch.Tensor):
        B, N = input_ids.shape
        y = self.y_init.view(1, 1, -1).expand(B, N, -1) # [B, N, C]
        z = self.z_init.view(1, 1, -1).expand(B, N, -1)
        return y, z

    def forward(self, input_ids: torch.Tensor, attention_mask = None):
        if attention_mask is not None:
            raise NotImplementedError("Attention mask is not supported for TRM.")

        y, z = self.init_y_z(input_ids)

        for _ in range(self.trm_N_steps):
            x = self.token_emb(input_ids) # [B, N, C]
            (y, z), cls_logits, _ = self.deep_recursion(x, y, z, self.trm_n, self.trm_T)

        return cls_logits

#======================================================================#
class TRMBatchLossfun:
    def __init__(self, N_steps: int = 10, n: int = 6, T: int = 3) -> None:
        self.N_steps, self.n, self.T = N_steps, n, T
        self.step_counter = 0
        self.batch = None
        self.y, self.z = None, None

    def __call__(self, trainer: mlutils.Trainer, model: TRMWrapper, batch: dict):

        #--------------------------------#
        # Bookeeping
        #--------------------------------#

        # get model.module if DDP
        model = mlutils.get_module(model)

        # Use each batch for N_steps. Dataloader guarantees each batch is repeated N_steps times
        FIRST_SUPERVISION_STEP = self.step_counter % self.N_steps == 0
        LAST_SUPERVISION_STEP = (self.step_counter + 1) % self.N_steps == 0
        assert trainer.repeat_train_batch == self.N_steps, f"Repeat train batch {trainer.repeat_train_batch} must be equal to N_steps {self.N_steps}. If that is not possible, consider caching the batch for the next steps. For reference, see the implementation in this commit: https://github.com/vpuri3/FLARE-dev.py/commit/2779a07615077bbbe08403d2ea060fa80a30065f."

        # Initialize y, z for each batch
        input_ids, labels = batch['input_ids'], batch['labels'] # [B, N]
        y, z = model.init_y_z(input_ids) if FIRST_SUPERVISION_STEP else (self.y, self.z) # [B, N, C]

        #--------------------------------#
        # Do current step
        #--------------------------------#

        # classification loss
        x = model.token_emb(input_ids) # [B, N, C]
        (y, z), cls_logits, halt_logits = model.deep_recursion(x, y, z, self.n, self.T)
        loss = F.cross_entropy(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))

        # # halting loss
        # halt_logits = halt_logits.view(-1) # [B]
        # halt_target = (cls_logits.argmax(dim=-1) == labels).all(dim=1) # [B]
        # loss += F.binary_cross_entropy_with_logits(halt_logits, halt_target.float())

        # # do halting
        # with torch.no_grad():
        #     halted = halt_target.bool()
        #     if LAST_SUPERVISION_STEP:
        #         halted[:] = True

        # if halt_logits.all().item() > 0:
        #     # reset counter to start new step
        #     self.step_counter = 0

        #--------------------------------#
        # Bookkeeping
        #--------------------------------#
        self.step_counter += 1
        self.y, self.z = (None, None) if LAST_SUPERVISION_STEP else (y, z)

        #--------------------------------#

        return loss

#======================================================================#
#