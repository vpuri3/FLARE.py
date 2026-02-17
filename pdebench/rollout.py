#
import torch
from torch import nn

import mlutils

__all__ = [
    'rollout',
]

#======================================================================#
@torch.no_grad()
def rollout(
    model,
    case_data,
    transform,
    verbose=False,
    device=None,
    init_step=0,
):

    if device is None:
        device = mlutils.select_device(device)
    model.to(device)

    nf = 2

    eval_data = []
    l2s = []
    r2s = []

    out_scale = transform.out_scale.to(device)
    out_shift = transform.out_shift.to(device)
    vel_shift = transform.vel_shift.to(device)
    vel_scale = transform.vel_scale.to(device)

    for (istep, data) in enumerate(case_data):
        data = data.clone()
        data.y = transform.make_fields(data, init_step + istep) # normalized velocity field
        eval_data.append(data)
        l2s.append(0.)
        r2s.append(1.)

    for k in range(1, len(eval_data)):
        _data = eval_data[k-1].clone().to(device) # given (k-1)-th step
        data  = eval_data[k  ].to(device)         # predict k-th step

        _data.x[:, -nf:] = _data.y[:, -nf:]

        delta = model(_data)
        delta = (delta * out_scale + out_shift) / vel_scale
        target = transform.make_fields(data, k + init_step) # normalized velocity field

        data.y = _data.x[:, -nf:] + delta
        data.e = data.y - target

        l2s[k] = nn.MSELoss()(data.e, 0 * data.e).item()
        r2s[k] = mlutils.r2(data.y, target)

        if verbose:
            print(f'Step {k}: {l2s[k], r2s[k]}')

        del _data

    del out_scale, out_shift, vel_shift, vel_scale

    return eval_data, l2s, r2s

#======================================================================#
#