"""
Plot energy spectrum of darcy or navier-stokes dataset solution
"""

import os
import sys
import socket
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set DATADIR_BASE based on machine
MACHINE = socket.gethostname()
if MACHINE == "eagle":
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')

# Output paths will be set based on dataset choice

#======================================================================#
def compute_energy_spectrum(field_2d):
    """
    Compute the energy spectrum (power spectrum) of a 2D field.

    Args:
        field_2d: 2D numpy array or torch tensor

    Returns:
        k: wavenumber magnitudes
        E: energy spectrum
    """
    # Convert to numpy if needed
    if torch.is_tensor(field_2d):
        field_2d = field_2d.numpy()

    # Get dimensions
    nx, ny = field_2d.shape

    # Compute 2D FFT
    fft_2d = np.fft.fft2(field_2d)
    fft_2d = np.fft.fftshift(fft_2d)  # Shift zero frequency to center

    # Compute power spectrum (magnitude squared)
    power = np.abs(fft_2d)**2

    # Create wavenumber arrays
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))

    # Create 2D wavenumber grid
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # Bin the power spectrum by wavenumber magnitude
    k_max = np.max(K)
    k_bins = np.linspace(0, k_max, 100)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    # Compute energy spectrum by averaging power in each bin
    E = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if np.any(mask):
            E[i] = np.mean(power[mask])

    return k_centers, E

#======================================================================#
def load_darcy_data():
    """Load darcy dataset and return first snapshot."""
    DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'darcy')

    # LNO configuration
    train_path = os.path.join(DATADIR, 'piececonst_r241_N1024_smooth1.mat')
    r = 1  # downsample
    h = int(((241 - 1) / r) + 1)
    s = h

    # Load training data
    train_data = scio.loadmat(train_path)
    y_train = train_data['sol'][:1, ::r, ::r][:, :s, :s]  # Get first snapshot only
    y_train = torch.from_numpy(y_train)

    # Get first snapshot and reshape to 2D
    solution_2d = y_train[0].numpy()  # Shape: (s, s) = (241, 241)

    return solution_2d, 'Darcy Solution'

#======================================================================#
def load_navier_stokes_data():
    """Load navier-stokes dataset and return last timestep of first trajectory."""
    DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'navier_stokes')
    # data_path = os.path.join(DATADIR, 'NavierStokes_V1e-5_N1200_T20.mat')
    data_path = os.path.join(DATADIR, 'ns_data_V1e-4_N20_T50_R256test.mat')

    # Load data
    data = scio.loadmat(data_path)['u']
    print(data.shape)
    solution_2d = data[10, :, :, -1] # 10, 12, 13

    return solution_2d, 'Navier-Stokes Solution'

#======================================================================#
def plot_spectrum(dataset='darcy'):
    """
    Plot energy spectrum for specified dataset.

    Args:
        dataset: 'darcy' or 'navier'
    """
    # Load data based on dataset choice
    if dataset == 'darcy':
        solution_2d, title = load_darcy_data()
        out_png = os.path.join(PROJDIR, 'figs', 'darcy_spectrum.png')
        out_pdf = os.path.join(PROJDIR, 'figs', 'darcy_spectrum.pdf')
    elif dataset == 'navier':
        solution_2d, title = load_navier_stokes_data()
        out_png = os.path.join(PROJDIR, 'figs', 'navier_spectrum.png')
        out_pdf = os.path.join(PROJDIR, 'figs', 'navier_spectrum.pdf')
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'darcy' or 'navier'")

    # Compute energy spectrum
    k, E = compute_energy_spectrum(solution_2d)

    # Normalize energy spectrum to percentage
    total_energy = np.sum(E)
    E_normalized = (E / total_energy) * 100  # Convert to percentage

    # Set matplotlib to use LaTeX fonts if available, otherwise use default
    try:
        import subprocess
        subprocess.run(['latex', '--version'], capture_output=True, check=True)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        print("Using LaTeX for plot rendering")
    except (subprocess.CalledProcessError, FileNotFoundError):
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]
        })
        print("LaTeX not available, using default matplotlib fonts")

    fontsize = 20

    # Create figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: 2D solution field
    # Set extent to map data to (-1, 1) range
    im1 = ax1.imshow(solution_2d, cmap='Spectral_r', origin='lower', aspect='auto',
                     extent=[-1, 1, -1, 1])
    # Add contour lines for better visibility
    x_coords = np.linspace(-1, 1, solution_2d.shape[1])
    y_coords = np.linspace(-1, 1, solution_2d.shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    # Create contour levels based on data range
    vmin, vmax = solution_2d.min(), solution_2d.max()
    n_levels = 10
    levels = np.linspace(vmin, vmax, n_levels)
    contours = ax1.contour(X, Y, solution_2d, levels=levels, colors='black', 
                           linewidths=0.5, alpha=0.4)
    ax1.set_title(title, fontsize=fontsize)
    ax1.set_xlabel('x', fontsize=fontsize)
    ax1.set_ylabel('y', fontsize=fontsize)
    ax1.set_xticks([-1, 0, 1])
    ax1.set_yticks([-1, 0, 1])
    cbar1 = plt.colorbar(im1, ax=ax1, label='Solution value')
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar1.set_label('Solution value', fontsize=fontsize)

    # Plot 2: Energy spectrum (normalized to percentage)
    ax2.loglog(k, E_normalized, 'b-', linewidth=2, label='Energy Spectrum')
    ax2.axhline(y=1e-7, color='r', linestyle='--', linewidth=2, label='FP32 Precision')
    ax2.set_xlabel('Wavenumber $k$', fontsize=fontsize)
    ax2.set_ylabel('Energy Spectrum $E(k)$ (\%)', fontsize=fontsize)
    ax2.set_title('Energy Spectrum', fontsize=fontsize)
    ax2.set_ylim(top=100)
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend(loc='upper right', fontsize=fontsize-2)

    # Increase tick label size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()

    # Save figures
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved file: {out_png}")
    print(f"Saved file: {out_pdf}")

    return

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot energy spectrum for darcy or navier-stokes dataset')
    parser.add_argument('--dataset', type=str, choices=['darcy', 'navier'], default='darcy',
                        help='Dataset to plot: darcy or navier (default: darcy)')
    args = parser.parse_args()

    plot_spectrum(dataset=args.dataset)

#======================================================================#
#