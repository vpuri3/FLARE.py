from __future__ import annotations

import numpy as np
import torch

from pdebench.dataset.plaid_elpl_v3.constants import CACHE_SCHEMA_VERSION
from pdebench.dataset.plaid_elpl_v3.schema import ShardPayload, TrajectoryBundle


def _dummy_traj(sim_id: int = 0) -> TrajectoryBundle:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
    cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
    sdf = torch.tensor([[1.0], [2.0], [3.0]])
    proj = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    u_traj = torch.randn(41, 3, 2)
    times = torch.linspace(0.0, 0.04, 41)
    return TrajectoryBundle(
        sim_id=sim_id,
        pos=pos,
        cells=cells,
        sdf=sdf,
        proj=proj,
        u_traj=u_traj,
        times=times,
        boundary_ids=torch.tensor([0, 1], dtype=torch.long),
        boundary_tags=("wall",),
        timestep_list=[float(v) for v in times.tolist()],
    )


def test_shard_round_trip() -> None:
    traj = _dummy_traj(sim_id=3)
    shard = ShardPayload(
        schema_version=CACHE_SCHEMA_VERSION,
        shard_id=1,
        sim_ids=[3],
        trajectories=[traj],
        laplacian=None,
    )
    restored = ShardPayload.from_dict(shard.to_dict())
    assert restored.shard_id == 1
    assert restored.sim_ids == [3]
    assert restored.trajectories[0].u_traj.shape == (41, 3, 2)
    np.testing.assert_allclose(restored.trajectories[0].pos.numpy(), traj.pos.numpy())
