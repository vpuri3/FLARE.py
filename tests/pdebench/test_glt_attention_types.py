import pytest
import torch

from pdebench.models.graph_models.glt import GLT, GLTBlock, GLTConfig, GLTMHAAttention
from pdebench.models.graph_models.glt import NonePEConfig, RawEigenPEConfig


def test_glt_config_defaults() -> None:
    cfg = GLTConfig()
    assert cfg.pe_inject_mode == "concat_input"
    assert cfg.pe_update is False
    assert isinstance(cfg.pe, RawEigenPEConfig)
    assert cfg.pe.kind == "raw_eigen"
    assert not hasattr(cfg, "glt_mode")


def test_glt_config_defaults_to_mha_attention() -> None:
    cfg = GLTConfig()
    assert cfg.attn_type == "mha"
    assert cfg.mlp_ratio == 4.0


def test_glt_rejects_unknown_inject_mode() -> None:
    with pytest.raises(ValueError, match="pe_inject_mode"):
        GLT(GLTConfig(pe_inject_mode="concat_qk_bad"), metadata=dict(c_in=3, c_out=1))


def test_glt_pe_update_requires_concat_qk() -> None:
    with pytest.raises(ValueError, match="pe_update"):
        GLT(
            GLTConfig(pe_inject_mode="concat_input", pe_update=True),
            metadata=dict(c_in=3, c_out=1),
        )


def test_glt_concat_input_none_pe_allowed() -> None:
    model = GLT(
        GLTConfig(pe_inject_mode="concat_input", pe=NonePEConfig()),
        metadata=dict(c_in=3, c_out=1),
    )
    assert model.pe.out_dim == 0


def test_glt_concat_qk_rejects_none_pe():
    with pytest.raises(ValueError, match="out_dim"):
        GLT(
            GLTConfig(pe_inject_mode="concat_qk", pe=NonePEConfig()),
            metadata=dict(c_in=3, c_out=1),
        )


def test_glt_block_concat_qk_qk_in_dim() -> None:
    block = GLTBlock(
        channel_dim=32,
        num_heads=4,
        pe_inject_mode="concat_qk",
        mlp_ratio=2.0,
        act="gelu",
        attn_type="mha",
    )
    assert block.attn.q_proj.in_features == 64


@pytest.mark.parametrize(
    ("attn_type", "expected_name"),
    [
        ("mha", "GLTMHAAttention"),
        ("linear", "GLTLinearAttention"),
        ("flare8", "GLTFlareAttention"),
    ],
)
def test_glt_block_constructs_attention_type(attn_type: str, expected_name: str) -> None:
    block = GLTBlock(
        channel_dim=32,
        num_heads=4,
        pe_inject_mode="concat_input",
        mlp_ratio=2.0,
        act="gelu",
        attn_type=attn_type,
    )

    assert type(block.attn).__name__ == expected_name


def test_glt_flare_attention_preserves_glt_qkv_projection_interface() -> None:
    block = GLTBlock(
        channel_dim=32,
        num_heads=4,
        pe_inject_mode="concat_qk",
        mlp_ratio=2.0,
        act="gelu",
        attn_type="flare8",
    )

    assert isinstance(block.attn, GLTMHAAttention)
    assert block.attn.q_proj is not block.attn.k_proj
    assert block.attn.v_proj.in_features == 32
    assert block.attn.q_proj.in_features == 64
    assert block.attn.k_proj.in_features == 64
    assert block.attn.num_latents == 8
    assert block.attn.separate_qk is True
    assert block.attn.condition_latents is True
    assert hasattr(block.attn, "latent_cond_k_proj")
    assert hasattr(block.attn, "latent_q_router")
    assert hasattr(block.attn, "latent_k_decode_router")
    assert block.attn.alpha_enc.shape == (4,)
    assert block.attn.alpha_enc[0].item() == pytest.approx(0.2)


@pytest.mark.parametrize("attn_type", ["transolver4", "flare", "flare0", "bad"])
def test_glt_block_rejects_invalid_attention_type(attn_type: str) -> None:
    with pytest.raises(ValueError):
        GLTBlock(
            channel_dim=32,
            num_heads=4,
            pe_inject_mode="concat_input",
            mlp_ratio=2.0,
            act="gelu",
            attn_type=attn_type,
        )


def test_glt_linear_attention_forward_cpu() -> None:
    block = GLTBlock(
        channel_dim=32,
        num_heads=4,
        pe_inject_mode="concat_qk",
        mlp_ratio=2.0,
        act="gelu",
        attn_type="linear",
    )
    x = torch.randn(7, 32)
    c = torch.randn(7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    out = block(x, c, cu_seqlens=cu_seqlens, max_seqlen=4)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def _packed_chain_edges(lengths: list[int]) -> torch.Tensor:
    parts = []
    offset = 0
    for length in lengths:
        if length > 1:
            src = torch.arange(offset, offset + length - 1, dtype=torch.long)
            dst = src + 1
            parts.append(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0))
        offset += length
    return torch.cat(parts, dim=1) if parts else torch.zeros(2, 0, dtype=torch.long)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GLT compile smoke test")
def test_glt_concat_input_k32_raw_torch_compile_cuda() -> None:
    lengths = [10, 12, 8, 15, 9, 11, 14, 10]
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.long).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    num_nodes = int(cu_seqlens[-1].item())
    batch_index = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.long),
        torch.tensor(lengths, device="cuda", dtype=torch.long),
    )
    pos = torch.randn(num_nodes, 2, device="cuda")
    edge_index = _packed_chain_edges(lengths).cuda()
    topology_features = torch.randn(num_nodes, 32, device="cuda")

    model = GLT(
        GLTConfig(
            pe_inject_mode="concat_input",
            pe=RawEigenPEConfig(num_eigenmodes=32),
            channel_dim=64,
            num_blocks=2,
            num_heads=4,
        ),
        metadata=dict(c_in=2, c_out=1),
    ).cuda()
    model = torch.compile(model)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            pos=pos,
            edge_index=edge_index,
            batch_index=batch_index,
            use_flash_varlen=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(lengths),
            topology_features=topology_features,
        )
        loss = out.square().mean()
    loss.backward()

    assert out.shape == (num_nodes, 1)
    assert torch.isfinite(out).all()
    assert torch.isfinite(loss).all()
