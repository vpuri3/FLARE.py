#
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'ModelWrapper',
]

from .backends import MODEL_TYPES
from .embeddings import TokenEmb, PosEmb, RotaryPositionalEmbeddings

#======================================================================#
class ModelWrapper(nn.Module):
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
        num_blocks: int = 4,
        backend: str = 'transformer',
        channel_dim: int = 128,
        num_heads: int = 4,
        act: str = None,
        rmsnorm: bool = False,
        **backend_kwargs,
    ):
        super().__init__()
        self.task = task
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.pos_embed_type = pos_embed

        #--------------------------------#
        # Pooling strategy
        #--------------------------------#
        if self.task in [
                'sudoku', 'match2', 'match3',
                'binary_relation_composition', 'quotient_binary_relation_composition'
            ]:
            self.pool = None
        else:
            assert pool in ['mean', 'max', 'cls']
            self.pool = pool

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(channel_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            max_length = max_length + 1
            if 'seq_len' in backend_kwargs:
                backend_kwargs['seq_len'] = backend_kwargs['seq_len'] + 1

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
        norm_after_pool = self.pool in ['mean', 'max']
        self.final_norm = Norm(channel_dim) if norm_after_pool else nn.Identity()

        cls_dim = channel_dim * 2 if self.task == 'retrieval' else channel_dim

        if task in ['image', 'pathfinder32', 'text', 'pathfinder128']:
            cls_proj = nn.Sequential(
                nn.Linear(cls_dim, cls_dim),
                nn.GELU(),
                nn.Linear(cls_dim, num_labels),
            )
        else: # listops, retrieval, mta-toy
            cls_proj = nn.Linear(cls_dim, num_labels)

        self.cls_proj = nn.Sequential(
            nn.Dropout(cls_drop),
            Norm(cls_dim) if norm_after_pool else nn.Identity(),
            cls_proj,
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

    def forward(self, input_ids, attention_mask = None):

        B, N = input_ids.shape
        device = input_ids.device
        original_B = B  # Save for final reshape if retrieval

        #--------------------------------#
        # Reshape input_ids for retrieval
        #--------------------------------#
        if self.task == 'retrieval':
            assert N == 2 * self.max_length, f"Sequence length must be 2 * max_length for retrieval. Got {N} and {self.max_length}."
            input_ids = input_ids.reshape(2 * B, self.max_length)
            B, N = input_ids.shape  # Update B and N after reshape

        #--------------------------------#
        # Get token embeddings
        #--------------------------------#
        x = self.token_emb(input_ids)  # [B, N, C]

        #--------------------------------#
        # Prepend CLS token if using CLS pooling
        #--------------------------------#
        if self.pool == 'cls':
            cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # [B, 1, C]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, C]
            N = N + 1
            # Prepend mask for CLS token (always valid/attended)
            if attention_mask is not None:
                cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [B, N+1]
        # Update B and N from actual tensor shape
        B, N = x.shape[:2]

        #--------------------------------#
        # Get positional embeddings for abs/sin (rope handled separately in blocks)
        #--------------------------------#
        if self.pos_emb is not None: 
            pos = self.pos_emb(B, N, device)
            pos = pos.unsqueeze(0).expand(B, -1, -1) if pos.dim() == 2 else pos # [B, N, C]
        else:
            pos = None

        #--------------------------------#
        # Process blocks
        #--------------------------------#
        for block in self.blocks:
            x = (x + pos) if pos is not None else x
            x = block(x, attention_mask=attention_mask)

        #--------------------------------#
        # Apply final normalization
        #--------------------------------#
        x = self.final_norm(x)

        #--------------------------------#
        # Pooling
        #--------------------------------#
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1).values
        elif self.pool == 'cls':
            x = x[:, 0]  # Take CLS token (first token)

        #--------------------------------#
        # Reshape for retrieval (concatenate the two sequence representations)
        #--------------------------------#
        if self.task == 'retrieval':
            x = x.reshape(original_B, -1)  # [original_B, 2*C]

        #--------------------------------#
        # Classify
        #--------------------------------#
        logits = self.cls_proj(x)

        return logits

#======================================================================#
#