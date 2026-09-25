import torch
from torch import nn

from pdebench.models.graph_models.gito import (
    GITOConfig,
    GITOHGTBlock,
    GITOLinearAttention,
    GITOModel,
    GITOTNOBlock,
)


def _looped_linear_attention(
    attn: GITOLinearAttention,
    query: torch.Tensor,
    context: torch.Tensor,
    query_batch: torch.Tensor,
    context_batch: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(query)
    for graph_idx in torch.unique(query_batch, sorted=True):
        q_idx = torch.nonzero(query_batch == graph_idx, as_tuple=False).flatten()
        c_idx = torch.nonzero(context_batch == graph_idx, as_tuple=False).flatten()
        out[q_idx] = attn._attend_one(query[q_idx], context[c_idx])
    return out


def test_gito_uses_hgt_then_tno_linear_attention() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=2, num_blocks_self_attn=3, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})

    assert len(model.hgt_blocks) == 2
    assert len(model.tno_blocks) == 3
    assert all(isinstance(block, GITOHGTBlock) for block in model.hgt_blocks)
    assert all(isinstance(block, GITOTNOBlock) for block in model.tno_blocks)
    assert all(isinstance(block.global_attn, GITOLinearAttention) for block in model.hgt_blocks)
    assert all(isinstance(block.fusion_attn, GITOLinearAttention) for block in model.hgt_blocks)
    assert all(isinstance(block.cross_attn, GITOLinearAttention) for block in model.tno_blocks)
    assert all(isinstance(block.self_attn, GITOLinearAttention) for block in model.tno_blocks)
    assert isinstance(model.node_encoder, nn.Linear)
    assert isinstance(model.edge_encoder, nn.Linear)
    assert isinstance(model.node_decoder, nn.Linear)


def test_gito_tno_has_cross_and_self_attention() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=1, num_blocks_self_attn=1, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})
    block = model.tno_blocks[0]
    assert hasattr(block, "cross_attn")
    assert hasattr(block, "self_attn")


def test_gito_linear_attention_batched_matches_looped_self_attention() -> None:
    torch.manual_seed(0)
    attn = GITOLinearAttention(hidden_dim=32, num_heads=4)
    query = torch.randn(7, 32)
    query_batch = torch.tensor([5, 2, 5, 2, 9, 9, 5], dtype=torch.long)

    actual = attn(query, query_batch=query_batch)
    expected = _looped_linear_attention(attn, query, query, query_batch, query_batch)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gito_linear_attention_batched_matches_looped_cross_attention() -> None:
    torch.manual_seed(0)
    attn = GITOLinearAttention(hidden_dim=32, num_heads=4)
    query = torch.randn(5, 32)
    context = torch.randn(8, 32)
    query_batch = torch.tensor([4, 4, 7, 9, 7], dtype=torch.long)
    context_batch = torch.tensor([9, 7, 4, 4, 7, 9, 7, 4], dtype=torch.long)

    actual = attn(query, context=context, query_batch=query_batch, context_batch=context_batch)
    expected = _looped_linear_attention(attn, query, context, query_batch, context_batch)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gito_forward_supports_batched_mesh_graphs() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=1, num_blocks_self_attn=1, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})
    pos = torch.randn(8, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
            [1, 2, 3, 0, 5, 6, 7, 4, 2, 0, 6, 4],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.shape[1], 4)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    out = model(pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch_index=batch_index)

    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_gito_forward_supports_zero_hgt_blocks_without_edges() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=0, num_blocks_self_attn=2, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})
    pos = torch.randn(8, 3)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    out = model(pos=pos, batch_index=batch_index)

    assert len(model.hgt_blocks) == 0
    assert len(model.tno_blocks) == 2
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_gito_forward_supports_zero_self_attention_blocks() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=2, num_blocks_self_attn=0, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})
    pos = torch.randn(8, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
            [1, 2, 3, 0, 5, 6, 7, 4, 2, 0, 6, 4],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.shape[1], 4)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    out = model(pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch_index=batch_index)

    assert len(model.hgt_blocks) == 2
    assert len(model.tno_blocks) == 0
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_gito_forward_supports_all_operator_blocks_disabled() -> None:
    cfg = GITOConfig(channel_dim=32, num_blocks_hgt=0, num_blocks_self_attn=0, num_heads=4)
    model = GITOModel(cfg, metadata={"c_in": 3, "c_out": 1, "c_edge": 4})
    pos = torch.randn(8, 3)

    out = model(pos=pos)

    assert len(model.hgt_blocks) == 0
    assert len(model.tno_blocks) == 0
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()
