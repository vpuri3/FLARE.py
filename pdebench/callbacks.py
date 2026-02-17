#
import os
import json
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist

import gc
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlutils
import pdebench

__all__ = [
    'RelL2Callback',
    'ScoresCallback',
]

#======================================================================#
class RelL2Callback(mlutils.Callback):
    def __init__(self, case_dir: str, dataset: str, x_normalizer, y_normalizer):
        super().__init__(case_dir)
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.dataset = dataset

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):

        trainer.model.eval()
        device = trainer.device

        lossfun = pdebench.RelL2Loss()
        y_normalizer = self.y_normalizer.to(device)

        _N, _rel_error, _r2 = 0, 0., []
        N_, rel_error_, r2_ = 0, 0., []

        for batch in trainer._loader_:
            x, y = batch[0].to(device), batch[1].to(device)
            with trainer.auto_cast:
                yh = trainer.model(x)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            l = lossfun(yh,y)

            _n = trainer.get_batch_size(batch, trainer._loader_)
            _N += _n
            _rel_error += l.item() * _n
            r2val = mlutils.r2(yh, y)
            _r2.append(r2val)
            del x, y, yh

        for batch in trainer.loader_:
            x, y = batch[0].to(device), batch[1].to(device)
            with trainer.auto_cast:
                yh = trainer.model(x)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            l = lossfun(yh,y)

            n_ = trainer.get_batch_size(batch, trainer.loader_)
            N_ += n_
            rel_error_ += l.item() * n_
            r2val = mlutils.r2(yh, y)
            r2_.append(r2val)
            del x, y, yh

        _r2 = torch.tensor(_r2)
        r2_ = torch.tensor(r2_)

        if trainer.DDP:
            # relative error
            pre_ddp, post_ddp = [_rel_error, rel_error_, _N, N_], []
            for p in pre_ddp:
                p = torch.tensor(p, device=trainer.device)
                dist.all_reduce(p, dist.ReduceOp.SUM)
                post_ddp.append(p.item())
            _rel_error, rel_error_, _N, N_ = post_ddp

            # R-Squared
            _r2 = _r2.to(device)
            r2_ = r2_.to(device)

            _r2_list = [torch.zeros_like(_r2) for _ in range(dist.get_world_size())]
            r2_list_ = [torch.zeros_like(r2_) for _ in range(dist.get_world_size())]

            dist.all_gather(_r2_list, _r2)
            dist.all_gather(r2_list_, r2_)

            _r2 = torch.cat(_r2_list, dim=0)
            r2_ = torch.cat(r2_list_, dim=0)

        _rel_error /= _N
        rel_error_ /= N_
        
        # save rel_error.json
        if trainer.GLOBAL_RANK == 0:
            print(f'Relative Error (train / test): {_rel_error:.8e} / {rel_error_:.8e}')
            with open(os.path.join(ckpt_dir, 'rel_error.json'), 'w') as f:
                json.dump({'train_rel_error': _rel_error, 'test_rel_error': rel_error_}, f)

            with open(os.path.join(ckpt_dir, '..', 'rel_error.json'), 'w') as f:
                json.dump({'train_rel_error': _rel_error, 'test_rel_error': rel_error_}, f)

        if trainer.GLOBAL_RANK == 0:
            print(f'Mean R2 (train / test): {_r2.mean():.4f} / {r2_.mean():.4f}')

        return

#======================================================================#
class ScoresCallback(mlutils.Callback):

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):

        trainer.model.eval()
        case_dir = self.case_dir

        assert trainer.WORLD_SIZE == 1, "ScoresCallback only supports single-rank evaluation"

        #--------------------------------#
        # get scores
        #--------------------------------#
        num_blocks = len(trainer.model.blocks)
        score_paths = [os.path.join(case_dir, 'scores', f'score_{l}.pt') for l in range(num_blocks)]

        if not all(os.path.exists(score_path) for score_path in score_paths):
            num_batches, MSE = 0, 0.0

            scores = [[] for _ in range(num_blocks)]

            for batch in trainer.loader_:
                x = batch[0].to(trainer.device)
                y = batch[1].to(trainer.device)
                yh, score = trainer.model(x, return_scores=True)

                n = trainer.get_batch_size(batch, trainer._loader_)
                num_batches += n
                MSE += ((yh - y).pow(2).mean() * n).item()
                for i in range(num_blocks):
                    scores[i].append(score[i].detach().cpu())

                del x, y, yh, score

            MSE = MSE / num_batches
            print()
            print(f"Train MSE: {MSE:.8e}")

            # save scores in case_dir/scores/score_<block_idx>.pt
            gc.collect()
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(case_dir, 'scores'), exist_ok=True)

            for l in range(num_blocks):
                scores_ = torch.cat(scores[l], dim=0)

                print(f"Saving scores to {os.path.join(case_dir, 'scores', f'score_{l}.pt')}")
                torch.save(scores_, os.path.join(case_dir, 'scores', f'score_{l}.pt'))

                del scores_
                scores[l] = None
                gc.collect()
                torch.cuda.empty_cache()

        scores = [torch.load(os.path.join(case_dir, 'scores', f'score_{l}.pt'), mmap=True, weights_only=True) for l in range(num_blocks)]
        gc.collect()
        torch.cuda.empty_cache()

        #--------------------------------#
        # attention weights
        #--------------------------------#
        num_batches, num_heads, num_latents, num_points = scores[0].shape
        eigen_paths = [os.path.join(case_dir, 'eigen', f'eigen_{l}.pt') for l in range(num_blocks)]

        if not all(os.path.exists(eigen_path) for eigen_path in eigen_paths):
            eigenvals = [[] for _ in range(num_blocks)]
            eigenvecs = [[] for _ in range(num_blocks)]
            chunk_size = 2
            num_chunks = (num_batches + chunk_size - 1) // chunk_size

            for l in range(num_blocks):
                gc.collect()
                torch.cuda.empty_cache()

                for chunk_idx in tqdm(range(num_chunks), desc=f"Processing block {l}", ncols=80):
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, len(scores[l]))
                    S = scores[l][start:end].to(trainer.device)  # [B H M N]

                    S = S.clamp(min=-30, max=30)
                    A  = torch.exp(S)
                    rsum = A.sum(dim=-1) # [B H M]
                    csum = A.sum(dim=-2) # [B H N]
                    LN = torch.diag_embed(1. / csum) # [B H N N]
                    LM = torch.diag_embed(1. / rsum) # [B H M M]
                    LM_sqrt = torch.sqrt(LM)
                    LN_sqrt = torch.sqrt(LN)
                    B  = LM_sqrt @ A @ LN_sqrt
                    BBT = B @ B.mT # [B H M M]

                    if torch.isnan(BBT).any() or torch.isinf(BBT).any():
                        print(f"Block {l}: Matrix contains NaN or Inf!")
                        print(f"S: {S.isnan().sum().item()}, {S.isinf().sum().item()}")
                        print(f"A: {A.isnan().sum().item()}, {A.isinf().sum().item()}")
                        print(f"rsum: {rsum.isnan().sum().item()}, {rsum.isinf().sum().item()}")
                        print(f"csum: {csum.isnan().sum().item()}, {csum.isinf().sum().item()}")
                        print(f"LN: {LN.isnan().sum().item()}, {LN.isinf().sum().item()}")
                        print(f"LM: {LM.isnan().sum().item()}, {LM.isinf().sum().item()}")
                        print(f"LM_sqrt: {LM_sqrt.isnan().sum().item()}, {LM_sqrt.isinf().sum().item()}")
                        print(f"LN_sqrt: {LN_sqrt.isnan().sum().item()}, {LN_sqrt.isinf().sum().item()}")
                        exit()

                    # SVD and eig decomposition are the same for symmetric matrices
                    U, SigmaSq, _ = torch.linalg.svd(BBT)
                    SigmaMat = torch.diag_embed(torch.sqrt(SigmaSq))

                    eigvals = SigmaSq
                    eigvecs = LN_sqrt @ B.mT @ U @ SigmaMat

                    eigenvals[l].append(eigvals.detach().cpu())
                    eigenvecs[l].append(eigvecs.detach().cpu())

                    del S, A, rsum, csum, LN, LM, LM_sqrt, LN_sqrt, B, BBT, SigmaMat, U, SigmaSq
                    del eigvals, eigvecs

            # save eigenvalues and eigenvectors in case_dir/eigen/eigen_<block_idx>.pt
            gc.collect()
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(case_dir, 'eigen'), exist_ok=True)

            for l in range(num_blocks):
                eigenvals_ = torch.cat(eigenvals[l], dim=0)
                eigenvecs_ = torch.cat(eigenvecs[l], dim=0)

                print(f"Saving eigen(values/vectors) to {os.path.join(case_dir, 'eigen', f'eigen_{l}.pt')}")
                torch.save([eigenvals_, eigenvecs_], os.path.join(case_dir, 'eigen', f'eigen_{l}.pt'))
                
                del eigenvals_, eigenvecs_
                eigenvals[l] = None
                eigenvecs[l] = None
                gc.collect()
                torch.cuda.empty_cache()

        eigen = [torch.load(eigen_path, mmap=True, weights_only=True) for eigen_path in eigen_paths]

        gc.collect()
        torch.cuda.empty_cache()

        #--------------------------------#
        # plot spectra
        #--------------------------------#

        eigenvals_means = [eigenvals.mean(dim=0) for (eigenvals, _) in eigen]

        nrows = math.ceil(math.sqrt(num_blocks))
        ncols = math.ceil(num_blocks / nrows)
        cutoff = int(num_latents * 1.1)

        # Plot mean eigenvalues across training cases
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        fig.suptitle('Eigenvalues of different heads for each block')
        for block in range(num_blocks):
            ax = axes.flat[block] if num_blocks > 1 else axes
            for h in range(num_heads):
                ax.plot(eigenvals_means[block][h, :cutoff])
            ax.axvline(x=num_latents-1, color='red', linestyle='--', label=f'Number of  Clusters = {num_latents}')
            ax.axhline(y=torch.finfo(torch.float32).eps, color='black', linestyle='--', label='Float32 Precision')
            ax.set_title(f'Block {block+1}')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Magnitude')
            # ax.legend()
            ax.set_yscale('log')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'eigenvals.png'))
        plt.close()

        #--------------------------------#
        # production quality spectra plot
        #--------------------------------#

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
        fontsize = 28
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xscale('linear')
            ax.set_yscale('log', base=10)
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.set_ylim(1e-8, 2e-0)
            ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])

        ax1.set_yticklabels(['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1e-0'])
        ax2.set_yticklabels(['', '', '', '', '', '', '', '', ''])
        ax3.set_yticklabels(['', '', '', '', '', '', '', '', ''])

        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)

        ax1.set_ylabel(r'Eigenvalue magnitude', fontsize=fontsize)
        ax1.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)
        ax2.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)
        ax3.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)

        ax1.set_title(r'Block 1', fontsize=fontsize)
        ax2.set_title(r'Block 5', fontsize=fontsize)
        ax3.set_title(r'Block 8', fontsize=fontsize)
        
        linewidth = 2.5

        for block, ax in zip([0, 4, 7], [ax1, ax2, ax3]):
            for h in range(num_heads):
                ax.plot(eigenvals_means[block][h, :cutoff], linewidth=linewidth)

            ax.axvline(x=num_latents-1, color='red', linestyle='--', linewidth=linewidth, label=r'Number of latents = %d' % num_latents)
            ax.axhline(y=torch.finfo(torch.float32).eps, color='black', linestyle='--', linewidth=linewidth, label=r'Float32 Precision')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'spectra.pdf'))
        plt.close()

        # #--------------------------------#
        # # Plot eigenvalues for first 10 test cases
        # #--------------------------------#

        # for case_idx in range(min(10, num_batches)):
        #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        #     fig.suptitle(f'Eigenvalues of different heads for each block - Case {case_idx}')

        #     for block in range(num_blocks):
        #         ax = axes.flat[block] if num_blocks > 1 else axes
        #         for h in range(num_heads):
        #             ax.plot(eigen[block][0][case_idx, h, :cutoff])
        #         ax.axvline(x=num_latents-1, color='red', linestyle='--', label=f'Number of  Clusters = {num_latents}')
        #         ax.axhline(y=torch.finfo(torch.float32).eps, color='black', linestyle='--', label='Float32 Precision')
        #         ax.set_title(f'Block {block+1}')
        #         ax.set_xlabel('Eigenvalue Index')
        #         ax.set_ylabel('Magnitude')
        #         ax.set_yscale('log')
        #         # ax.legend()
        #         # ax.set_ylim(bottom=torch.finfo(torch.float32).eps)
        #         ax.grid(True)

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(ckpt_dir, f'eigenvals{case_idx}.png'))
        #     plt.close()

        # #--------------------------------#
        # # cosine similarity of eigenvalue spectra
        # #--------------------------------#

        # nrows = math.ceil(math.sqrt(num_blocks))
        # ncols = math.ceil(num_blocks / nrows)
        # cutoff = int(num_latents * 1.1)
        
        # # Plot mean eigenvalues across training cases
        # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        # fig.suptitle('Similarity of eigenvalue spectra among heads for each block')
        # for block in range(num_blocks):
        #     ax = axes.flat[block] if num_blocks > 1 else axes

        #     eigvals = eigenvals_means[block] # [H, N]
        #     eigvals = eigvals.abs() / eigvals.abs().sum(dim=-1, keepdim=True) # abs is unnecessary, but just in case

        #     # cosine similarity between heads
        #     similarity = []
        #     for h1 in range(num_heads):
        #         for h2 in range(h1+1, num_heads):
        #             cos_sim = F.cosine_similarity(eigvals[h1], eigvals[h2], dim=-1)
        #             similarity.append(cos_sim)

        #     # Create empty matrix and fill upper triangle (excluding diagonal)
        #     similarity_matrix = torch.zeros(num_heads, num_heads)
        #     triu_indices = torch.triu_indices(num_heads, num_heads, offset=1)
        #     similarity_matrix[triu_indices[0], triu_indices[1]] = torch.tensor(similarity)
        #     # Make matrix symmetric by copying upper triangle to lower
        #     similarity_matrix = similarity_matrix + similarity_matrix.T
        #     # set diagonal to 1
        #     similarity_matrix.fill_diagonal_(1.0)

        #     # set colorbar range to -1 to 1
        #     im = ax.imshow(similarity_matrix, cmap='gray', aspect='auto', vmin=-1, vmax=1)
        #     ax.set_title(f'Block {block+1}')
        #     ax.set_xlabel('Head Index')
        #     ax.set_ylabel('Head Index')
        #     ax.grid(True)

        #     cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_ticks([-1, 0, 1])
        #     cbar.set_ticklabels(['-1', '0', '1'])

        #     ax.set_xticks(range(num_heads))
        #     ax.set_yticks(range(num_heads))
        #     ax.set_xticklabels(range(num_heads))
        #     ax.set_yticklabels(range(num_heads))

        # plt.tight_layout()
        # plt.savefig(os.path.join(ckpt_dir, 'eigenvals_similarity.png'))
        # plt.close()


        #--------------------------------#
        # visualize eigenvectors for individual cases
        #--------------------------------#
        # for case_idx in range(min(10, num_batches)):
        #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        #     fig.suptitle(f'Eigenvectors of different heads for each block - Case {case_idx}')

        #     for block in range(L):
        #         ax = axes.flat[block] if L > 1 else axes
        #         for h in range(num_heads):
        #             ax.imshow(eigenvecs[block][case_idx, h, :, :])
        #             ax.set_title(f'Block {block+1}, Head {h}')
        #             ax.set_xlabel('Cluster Index')
        #             ax.set_ylabel('Point Index')
        #             ax.grid(True)

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(ckpt_dir, f'eigenvecs{case_idx}.png'))
        #     plt.close()

        #--------------------------------#
        # cosine similarity between eigenvectors for individual cases
        #--------------------------------#

        # cluster utilizaton
        # connection sparsity
        # cross-head correlation <-- how much do attention weights correlate across heads?
        # plot eigenvalue decay of W

        # # Attention sparsity: what proportion of attention weights are non-zero?
        # Att_sparsity = [(Att > 1e-2).sum(dim=-2).float().mean().item() * 100 for Att in Atts]

        # print(f"Att sparsity: {[round(s, 2) for s in Att_sparsity]}")

        # print(f"Cluster utilization: How evenly are clusters used?")
        # print(f"Want every cluster to be used equally often, avoiding scenarios where")
        # print(f"some clusters are always ignored (underutilized) or overly relied upon (overutilized).")
        # print(f"Sparsity: What proportion of clusters are invoked per point?")

        # # Cluster utilization (how evenly are clusters used)
        # target_use = 1 / M
        # threshold = 0.5 * target_use
        # cluster_use = [w.mean(dim=-1) for w in W_encodes] # [B H M]
        # underused = [(cluster_use[i] < (target_use - threshold)).float().mean().item() * 100 for i in range(B)]
        # overused = [(cluster_use[i] > (target_use + threshold)).float().mean().item() * 100 for i in range(B)]

        # print()
        # print(f"Cluster utilization stats:")
        # print(f"  Mean: {[round(s.mean().item(), 4) for s in cluster_use]} (Target: {target_use:.5f})")
        # print(f"  Std : {[round(s.std(dim=-1).mean().item(), 4) for s in cluster_use]}")
        # print(f"  Min : {[round(s.min().item(), 4) for s in cluster_use]}")
        # print(f"  Max : {[round(s.max().item(), 4) for s in cluster_use]}")
        # print(f"  % Underused (< {target_use - threshold:.5f}): {[round(u, 2) for u in underused]}. Mean: {sum(underused) / len(underused):.4f}")
        # print(f"  % Overused  (> {target_use + threshold:.5f}): {[round(o, 2) for o in overused]}. Mean: {sum(overused) / len(overused):.4f}")
        # print(f"  % Sparsity : {[round(s, 2) for s in Att_sparsity]}. Mean: {sum(Att_sparsity) / len(Att_sparsity):.4f}")
        # print()

        # mean_cluster_use = [w.mean(dim=[0,-1]) / target_use for w in W_encodes]

        # fig, axes = plt.subplots(B, 2, figsize=(10, 3*B))
        # fig.suptitle('Cluster Utilization')

        # for i in range(B):
        #     Att = Atts[i]
        #     ax = axes[i, 0]
        #     im = ax.imshow(Att[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=100)
        #     ax.set_title(f'Layer {i}: sparsity: {Att_sparsity[i]:.1f}%')
        #     ax.set_xlabel('Cluster Index')
        #     ax.set_ylabel('Head Index')
        #     fig.colorbar(im, ax=ax)
        #     im.cmap.set_over('red')
        #     im.cmap.set_under('blue')

        # plt.tight_layout()
        # plt.savefig(os.path.join(ckpt_dir, 'utilization.png'))
        # plt.savefig(os.path.join(ckpt_dir, '..', 'utilization.png'))

        return

#======================================================================#
def eig1(S):
    _, _, M, N = S.shape

    We = F.softmax(S, dim=-1) # sum over N
    Wd = F.softmax(S, dim=-2) # sum over M
    W = Wd.mT @ We

    # U, Sv, V = torch.linalg.svd(W, full_matrices=False)
    # eigvals = Sv[:,:,:M]
    # eigvecs = V[:,:,:M,:]

    eigvals, eigvecs = torch.linalg.eig(W) # [B H N], [B H N N]
    eigvals = eigvals[:,:,:M]
    eigvecs = eigvecs[:,:,:,:M]
    
    print(f'Eig1: eigvals: {eigvals.shape}, eigvecs: {eigvecs.shape}')

    return We, Wd, W, eigvals, eigvecs

def eig2(S):
    _, _, M, N = S.shape

    A  = torch.exp(S)
    rsum = A.sum(dim=-1) # [B H M]
    csum = A.sum(dim=-2) # [B H N]

    LN = torch.diag_embed(1. / csum) # [B H N N]
    LM = torch.diag_embed(1. / rsum) # [B H M M]

    We = LM @ A
    Wd = A @ LN

    W1 = Wd.mT @ We          # verified
    W2 = LN @ A.mT @ LM @ A  # verified

    LM_sqrt = torch.sqrt(LM)
    LN_sqrt = torch.sqrt(LN)
    LN_sqrt_inv = torch.diag_embed(1.0 / torch.diagonal(LN_sqrt, dim1=-2, dim2=-1))
    B  = LM_sqrt @ A @ LN_sqrt

    BTB = B.mT @ B
    W = LN_sqrt @ BTB @ LN_sqrt_inv

    tol = 1e-6
    assert (W1 - W).abs().max() < tol
    assert (W2 - W).abs().max() < tol

    # ### METHOD 1
    # U, Sv, V = torch.linalg.svd(B, full_matrices=False)
    # eigvals = Sv**2
    # eigvecs = (LN_sqrt @ V.mT).mT

    ### METHOD 2
    BBT = B @ B.mT # [B H M M]
    U, SigmaSq, _ = torch.linalg.svd(BBT) # SVD and eig decomposition are the same for symmetric matrices
    SigmaMat = torch.diag_embed(torch.sqrt(SigmaSq))

    eigvals = SigmaSq
    eigvecs = LN_sqrt @ B.mT @ U @ torch.linalg.inv(SigmaMat)

    print(f'Eig2: eigvals: {eigvals.shape}, eigvecs: {eigvecs.shape}')

    return We, Wd, W, eigvals, eigvecs

def main():
    B, H, M, N = 2, 4, 16, 50

    S = torch.rand(B, H, M, N)

    We1, Wd1, W1, eigvals1, eigvecs1 = eig1(S)
    We2, Wd2, W2, eigvals2, eigvecs2 = eig2(S)
    
    ranks1 = torch.linalg.matrix_rank(W1)
    ranks2 = torch.linalg.matrix_rank(W2)

    assert (ranks1 == M).all()
    assert (ranks2 == M).all()

    e1 = (We1 - We2).abs().max()
    e2 = (Wd1 - Wd2).abs().max()
    e3 = (W1  - W2 ).abs().max()
    e4 = (eigvals1 - eigvals2).norm(2) / eigvals1.numel()
    e5 = (eigvecs1 - eigvecs2).norm(2) / eigvecs1.numel()
    print(f"We: {e1:.4f}, Wd: {e2:.4f}, W: {e3:.4f}, eigvals: {e4:.4f}, eigvecs: {e5:.4f}")

if __name__ == "__main__":
    main()
#======================================================================#
#