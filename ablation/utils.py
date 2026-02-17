#======================================================================#
# Utils
#======================================================================#

import os
import shutil
import subprocess
import time
from tqdm import tqdm

__all__ = [
    'run_jobs'
]

#======================================================================#
def run_jobs(
    job_queue: list, gpu_count: int, max_jobs_per_gpu: int, reverse_queue: bool = False,
    dataset: str = 'elasticity', epochs: int = 500, batch_size: int = 2, weight_decay: float = 1e-5,
):
    jobid = 0
    njobs = len(job_queue)

    if reverse_queue:
        job_queue = job_queue[::-1]

    print(f"Running {njobs} jobs on {gpu_count} GPUs.")
    pbar = tqdm(total=njobs, desc="Running jobs", ncols=80)

    # Run jobs
    active_processes = [[] for _ in range(gpu_count)]
    while job_queue or any(len(p) > 0 for p in active_processes):

        # Check completed processes
        for i in range(gpu_count):
            # p.poll() returns None if the process is still running
            for p in active_processes[i]:
                if p.poll() is not None:
                    if p.returncode != 0:
                        print(f"\nExperiment {p.args[6]} failed on GPU {i}. Removing and re-running.")
                        # remove failed experiment and re-run
                        case_dir = os.path.join('.', 'out', 'pdebench', p.args[6])
                        shutil.rmtree(case_dir)
                        job_queue.append(job)
                        pbar.update(-1)
                        jobid -= 1

            # Remove completed processes
            active_processes[i] = [p for p in active_processes[i] if p.poll() is None]

        # Start new jobs on available GPUs
        while any(len(p) < max_jobs_per_gpu for p in active_processes) and job_queue:
            gpuid = min(range(gpu_count), key=lambda i: len(active_processes[i]))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

            job = job_queue.pop(0)
            model_type = job['model_type']
            num_layers_ffn = job['num_layers_ffn']
            process = subprocess.Popen(
                [
                    'uv', 'run', 'python', '-m', 'pdebench',
                    '--exp_name', str(job['exp_name']),
                    '--train', str('True'),
                    '--model_type', str(model_type),
                    '--dataset', str(dataset),
                    '--seed', str(job['seed']),
                    # training arguments
                    '--epochs', str(epochs),
                    '--weight_decay', str(weight_decay),
                    '--batch_size', str(batch_size),
                    # model arguments
                    '--channel_dim', str(job['channel_dim']),
                    '--num_latents', str(job['num_latents']),
                    '--num_blocks', str(job['num_blocks']),
                    '--num_heads', str(job['num_heads']),
                    '--num_layers_kv_proj', str(job['num_layers_kv_proj']),
                    '--num_layers_ffn', str(num_layers_ffn),
                    '--num_layers_in_out_proj', str(job['num_layers_in_out_proj']),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            active_processes[gpuid].append(process)
            jobid += 1

            pbar.update(1)

        time.sleep(60)
    return

#======================================================================#
#
