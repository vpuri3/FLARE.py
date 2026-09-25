#!/usr/bin/env python3
"""
Evaluation script for loopy experiments.
Plots num_blocks vs train/test relative error for different model variants.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


def parse_num_blocks(bch_dir_name):
    """Extract num_blocks from directory name like 'B_2' or 'B_2_P_20'"""
    # Try B_<NUM>_P_<NUM> pattern first (for loopy variants, effective num_blocks = num_blocks * num_passes)
    match = re.match(r'B_(\d+)_P_(\d+)$', bch_dir_name)
    if match:
        return int(match.group(1)) * int(match.group(2))
    # Try B_<NUM> pattern (for flare_C_* directories)
    match = re.match(r'B_(\d+)$', bch_dir_name)
    if match:
        return int(match.group(1))
    return None


def format_variant_label(variant_name):
    """Format variant name for legend display."""
    if variant_name.startswith('transformer'):
        # Parse transformer variant: transformer_C_128_H_4
        pattern = r'transformer_C_(\d+)_H_(\d+)$'
        match = re.match(pattern, variant_name)
        if match:
            C = int(match.group(1))
            H = int(match.group(2))
            return f'Transformer C={C}, H={H}'
        # Fallback for other transformer formats
        return 'Transformer'
    
    if variant_name.startswith('flare_tied_att_'):
        # Parse FLARE tied attention variant: flare_tied_att_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
        pattern = r'flare_tied_att_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_(\w+)_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
        match = re.match(pattern, variant_name)

        if match:
            C = int(match.group(1))
            H = int(match.group(2))
            M = int(match.group(3))
            scale = match.group(4)
            num_layers_kv_proj = int(match.group(5))
            num_layers_ffn = int(match.group(6))
            kv_proj_mlp_ratio = float(match.group(7))
            ffn_mlp_ratio = float(match.group(8))

            return f"FLARE (tied ATT) C={C}, H={H}, M={M}, scale={scale}, KV (L={num_layers_kv_proj}, r={kv_proj_mlp_ratio}), FFN (L={num_layers_ffn}, r={ffn_mlp_ratio})"

    if variant_name.startswith('flare_tied_ffn_'):
        # Parse FLARE tied FFN variant: flare_tied_ffn_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
        pattern = r'flare_tied_ffn_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_(\w+)_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
        match = re.match(pattern, variant_name)

        if match:
            C = int(match.group(1))
            H = int(match.group(2))
            M = int(match.group(3))
            scale = match.group(4)
            num_layers_kv_proj = int(match.group(5))
            num_layers_ffn = int(match.group(6))
            kv_proj_mlp_ratio = float(match.group(7))
            ffn_mlp_ratio = float(match.group(8))

            return f"FLARE (tied FFN) C={C}, H={H}, M={M}, scale={scale}, KV (L={num_layers_kv_proj}, r={kv_proj_mlp_ratio}), FFN (L={num_layers_ffn}, r={ffn_mlp_ratio})"

    if variant_name.startswith('flare_'):
        # Parse FLARE variant: flare_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
        pattern = r'flare_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_(\w+)_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
        match = re.match(pattern, variant_name)

        if match:
            C = int(match.group(1))
            H = int(match.group(2))
            M = int(match.group(3))
            scale = match.group(4)
            num_layers_kv_proj = int(match.group(5))
            num_layers_ffn = int(match.group(6))
            kv_proj_mlp_ratio = float(match.group(7))
            ffn_mlp_ratio = float(match.group(8))

            return f"FLARE C={C}, H={H}, M={M}, scale={scale}, KV (L={num_layers_kv_proj}, r={kv_proj_mlp_ratio}), FFN (L={num_layers_ffn}, r={ffn_mlp_ratio})"

    elif variant_name.startswith('loopy_'):
        # Parse LOOPY variant: loopy_B_2_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
        pattern = r'loopy_B_(\d+)_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_(\w+)_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
        match = re.match(pattern, variant_name)

        if match:
            B = int(match.group(1))
            C = int(match.group(2))
            H = int(match.group(3))
            M = int(match.group(4))
            scale = match.group(5)
            num_layers_kv_proj = int(match.group(6))
            num_layers_ffn = int(match.group(7))
            kv_proj_mlp_ratio = float(match.group(8))
            ffn_mlp_ratio = float(match.group(9))

            return f"FLARE (tied B={B} blocks) C={C}, H={H}, M={M}, scale={scale}, KV (L={num_layers_kv_proj}, r={kv_proj_mlp_ratio}), FFN (L={num_layers_ffn}, r={ffn_mlp_ratio})"

    # Fallback: return original name
    return variant_name


def collect_data(base_dir, dataset='elasticity'):
    """
    Collect relative error data from all checkpoint directories.
    
    Returns:
        dict: {variant_name: [(num_blocks, train_rel_error, test_rel_error), ...]}
    """
    data = defaultdict(list)
    dataset_dir = Path(base_dir) / f'loopy_{dataset}'

    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_dir} does not exist")
        return data

    # Iterate over all variant directories
    for variant_dir in sorted(dataset_dir.iterdir()):
        if not variant_dir.is_dir():
            continue

        variant_name = variant_dir.name

        # Iterate over all B_* / B_*_P_* subdirectories
        for b_dir in sorted(variant_dir.iterdir()):
            if not b_dir.is_dir():
                continue
            
            # Parse num_blocks from directory name (handles B_X, B_X_P_Y patterns)
            num_blocks = parse_num_blocks(b_dir.name)
            if num_blocks is None:
                continue
            
            # Check for rel_error.json in ckpt10
            rel_error_file = b_dir / 'ckpt10' / 'rel_error.json'
            
            if rel_error_file.exists():
                try:
                    with open(rel_error_file, 'r') as f:
                        rel_error_data = json.load(f)
                    
                    train_rel_error = rel_error_data.get('train_rel_error')
                    test_rel_error = rel_error_data.get('test_rel_error')
                    
                    if train_rel_error is not None and test_rel_error is not None:
                        data[variant_name].append((num_blocks, train_rel_error, test_rel_error))
                        print(f"Found: {variant_name}/{b_dir.name} - train: {train_rel_error:.6f}, test: {test_rel_error:.6f}")
                except Exception as e:
                    print(f"Error reading {rel_error_file}: {e}")
    
    return data


def get_variant_signature(variant_name):
    """Extract signature from variant name to match flare and loopy variants."""
    # Extract C, H, M, KVLAYERS, FFNLAYERS, KVRATIO, FFNRATIO from variant name
    # This signature will be used to match flare and loopy variants
    
    # Pattern for flare_tied_att: flare_tied_att_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
    flare_tied_att_pattern = r'flare_tied_att_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_\w+_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
    match = re.match(flare_tied_att_pattern, variant_name)
    if match:
        return tuple(match.groups())
    
    # Pattern for flare_tied_ffn: flare_tied_ffn_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
    flare_tied_ffn_pattern = r'flare_tied_ffn_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_\w+_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
    match = re.match(flare_tied_ffn_pattern, variant_name)
    if match:
        return tuple(match.groups())
    
    # Pattern for flare: flare_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
    flare_pattern = r'flare_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_\w+_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
    match = re.match(flare_pattern, variant_name)
    if match:
        return tuple(match.groups())
    
    # Pattern for loopy: loopy_B_2_C_128_H_4_M_256_ATTNSCALE_sqrt_KVLAYERS_3_FFNLAYERS_3_KVRATIO_1.0_FFNRATIO_1.0
    loopy_pattern = r'loopy_B_\d+_C_(\d+)_H_(\d+)_M_(\d+)_ATTNSCALE_\w+_KVLAYERS_(-?\d+)_FFNLAYERS_(-?\d+)_KVRATIO_([\d.]+)_FFNRATIO_([\d.]+)'
    match = re.match(loopy_pattern, variant_name)
    if match:
        return tuple(match.groups())
    
    # For transformer or other variants, use the full name as signature
    return variant_name


def get_marker_for_variant(variant_name):
    """Get marker style based on variant type and B value for LOOPY variants."""
    if variant_name.startswith('loopy_'):
        # Extract B value from LOOPY variant: loopy_B_2_C_128_H_4_...
        pattern = r'loopy_B_(\d+)_'
        match = re.match(pattern, variant_name)
        if match:
            B = int(match.group(1))
            # Map B values to markers
            marker_map = {
                1: 'o',  # circle
                2: 's',  # square
                4: '^',  # triangle
                8: 'v',  # triangle down
            }
            return marker_map.get(B, 'o')  # Default to circle if B not in map
    
    if variant_name.startswith('flare_tied_att_'):
        return 'D'  # diamond
    
    if variant_name.startswith('flare_tied_ffn_'):
        return '*'  # star
    
    # Default marker for non-LOOPY variants (transformer, flare, etc.)
    return 'o'  # circle


def plot_results(data, output_dir, dataset='elasticity'):
    """Create plots for train and test relative errors."""

    # Sort variants for consistent plotting
    variants = sorted(data.keys())
    # variants = [v for v in variants if not v.startswith('loopy')]

    if not variants:
        print("No data found to plot!")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Assign colors to variant signatures (matching flare and loopy get same color)
    signature_to_color = {}
    color_cycle = plt.cm.tab10.colors  # Use tab10 colormap
    color_idx = 0

    for variant in variants:
        signature = get_variant_signature(variant)
        if signature not in signature_to_color:
            signature_to_color[signature] = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1

    # Collect handles and labels for common legend
    handles = []
    labels = []

    # Plot train relative error
    for variant in variants:
        if not data[variant]:
            continue

        # Sort by num_blocks
        sorted_data = sorted(data[variant], key=lambda x: x[0])
        num_blocks = [x[0] for x in sorted_data]
        train_errors = [x[1] for x in sorted_data]
        
        # Format label for legend
        label = format_variant_label(variant)
        
        # Get color, line style, and marker
        signature = get_variant_signature(variant)
        color = signature_to_color[signature]
        if variant.startswith('loopy_'):
            linestyle = '--'  # dashed
        elif variant.startswith('flare_tied_att_'):
            linestyle = ':'  # dotted
        elif variant.startswith('flare_tied_ffn_'):
            linestyle = ':'  # dotted
        else:
            linestyle = '-'  # solid
        marker = get_marker_for_variant(variant)
        
        markersize = 12 if marker == '*' else 6
        
        line, = ax1.plot(num_blocks, train_errors, marker=marker, label=label, 
                        linewidth=1, markersize=markersize, color=color, linestyle=linestyle)
        handles.append(line)
        labels.append(label)
    
    ax1.set_xlabel('Effective Number of Blocks', fontsize=12)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.set_title(f'Train Relative Error ({dataset})', fontsize=14)
    # ax1.set_ylim(1e-3, 4e-1)
    y_ticks = ([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2] + 
               [2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1])
    y_tick_labels = (['1e-3', '2e-3', '', '4e-3', '', '6e-3', '', '8e-3', '', '1e-2'] + 
                     ['2e-2', '', '4e-2', '', '6e-2', '', '8e-2', '', '1e-1'])

    ax2.set_xlabel('Effective Number of Blocks', fontsize=12)
    ax2.set_title(f'Test Relative Error ({dataset})', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        # ax2.set_ylim(1e-3, 1e-1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    
    ax1.set_ylim(9e-4, 5e-2)
    ax2.set_ylim(3e-3, 5e-2)

    # Plot test relative error (reuse same colors and line styles)
    for variant in variants:
        if not data[variant]:
            continue
        
        # Sort by num_blocks
        sorted_data = sorted(data[variant], key=lambda x: x[0])
        num_blocks = [x[0] for x in sorted_data]
        test_errors = [x[2] for x in sorted_data]
        
        # Get color, line style, and marker (same as train plot)
        signature = get_variant_signature(variant)
        color = signature_to_color[signature]
        if variant.startswith('loopy_'):
            linestyle = '--'  # dashed
        elif variant.startswith('flare_tied_att_'):
            linestyle = ':'  # dotted
        elif variant.startswith('flare_tied_ffn_'):
            linestyle = ':'  # dotted
        else:
            linestyle = '-'  # solid

        marker = get_marker_for_variant(variant)
        markersize = 12 if marker == '*' else 6
        
        ax2.plot(num_blocks, test_errors, marker=marker, linewidth=1, markersize=markersize, 
                color=color, linestyle=linestyle)
    
    # Create common legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12, 
               bbox_to_anchor=(0.5, -0.5), frameon=True)
    
    # Adjust layout to make room for bottom legend
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # [left, bottom, right, top] - reserve 15% at bottom
    
    # Save plots
    output_path = Path(output_dir) / f'loopy_{dataset}_eval.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = Path(output_dir) / f'loopy_{dataset}_eval.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plots saved to: {output_path_pdf}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate loopy experiments')
    parser.add_argument('--dataset', type=str, default='elasticity',
                        help='Dataset name (e.g., elasticity)')
    
    args = parser.parse_args()
    base_dir = 'out/pdebench'
    output_dir = f'out/pdebench/loopy_{args.dataset}'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting data from {base_dir}/loopy_{args.dataset}...")
    data = collect_data(base_dir, args.dataset)
    
    if not data:
        print("No data found!")
        return
    
    print(f"\nFound data for {len(data)} variants:")
    for variant in sorted(data.keys()):
        print(f"  {variant}: {len(data[variant])} experiments")
    
    print("\nGenerating plots...")
    plot_results(data, output_dir, args.dataset)

if __name__ == '__main__':
    main()

