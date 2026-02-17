"""
decoder_next_token_matplotlib.py

Usage:
    python decoder_next_token_matplotlib.py
    # -> creates decoder_next_token.gif in the same folder
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------- config ---------------------- #
N_TOKENS = 8
N_LATENT = 3

INPUT_INACTIVE_COLOR = "#dfffd4"  # light green-ish
OUTPUT_INACTIVE_COLOR = "#c7dfff" # light blue-ish

LATENT_INACTIVE_COLOR = "#ffd4e5" # light pink-ish
LATENT_ACTIVE_COLOR = "#d455a8"  # darker pink

INPUT_ACTIVE_COLOR = "#55a868"    # darker green
OUTPUT_ACTIVE_COLOR = "#4c72b0"   # darker blue

EDGE_COLOR = "gray"
EDGE_ALPHA_ACTIVE = 0.8
EDGE_ALPHA_INACTIVE = 0.0

FPS = 1          # frames per second for GIF
INTERVAL = 500   # ms between frames in the live preview

# High resolution settings
DPI = 200        # High DPI for crisp output
SCALE = 2.0      # Scale factor for high resolution (multiplies figure size and linewidths)

# ---------------------- setup latex ---------------------- #
# Set matplotlib to use LaTeX fonts
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

# ---------------------- setup figure ---------------------- #
fig, ax = plt.subplots(figsize=(5 * SCALE, 3 * SCALE), dpi=DPI)

# token positions
x_positions = np.arange(N_TOKENS) * 0.5
# Center latent nodes horizontally within the input/output span
x_min, x_max = x_positions[0], x_positions[-1]
if N_LATENT > 1:
    center_latent = (x_min + x_max) / 2
    x_dist = x_positions[1] - x_positions[0]
    x_latent_positions = center_latent + np.arange(-(N_LATENT - 1) / 2, (N_LATENT - 1) / 2 + 1) * x_dist
else:
    x_latent_positions = np.array([(x_min + x_max) / 2])

y_input = 2.0
y_latent = 1.0
y_output = 0.0

input_patches = []
latent_patches = []
output_patches = []

# Create input, latent, and output token circles
# Scale visual elements for high resolution
CIRCLE_RADIUS = 0.06 * SCALE
CIRCLE_LINEWIDTH = 1.0 * SCALE

# Input nodes (skip last one: i < N_TOKENS - 1)
for i, x in enumerate(x_positions):
    if i < N_TOKENS - 1:
        c_in = Circle(
            (x, y_input),
            radius=CIRCLE_RADIUS,
            facecolor=INPUT_INACTIVE_COLOR,
            edgecolor="black",
            linewidth=CIRCLE_LINEWIDTH,
            zorder=3,
        )
        ax.add_patch(c_in)
        input_patches.append(c_in)
    else:
        input_patches.append(None)  # Placeholder for skipped token

# Latent nodes
for i, x in enumerate(x_latent_positions):
    c_latent = Circle(
        (x, y_latent),
        radius=CIRCLE_RADIUS,
        facecolor=LATENT_INACTIVE_COLOR,
        edgecolor="black",
        linewidth=CIRCLE_LINEWIDTH,
        zorder=3,
    )
    ax.add_patch(c_latent)
    latent_patches.append(c_latent)

# Output nodes (skip first one: i > 0)
for i, x in enumerate(x_positions):
    if i > 0:
        c_out = Circle(
            (x, y_output),
            radius=CIRCLE_RADIUS,
            facecolor=OUTPUT_INACTIVE_COLOR,
            edgecolor="black",
            linewidth=CIRCLE_LINEWIDTH,
            zorder=3,
        )
        ax.add_patch(c_out)
        output_patches.append(c_out)
    else:
        output_patches.append(None)  # Placeholder for skipped token

# Dashed bounding box (roughly similar to your draw.io figure)
padding = 0.6
xmin, xmax = x_positions[0] - padding, x_positions[-1] + padding
ymin, ymax = y_output - 0.4, y_input + 0.4

bbox = Rectangle(
    (xmin, ymin),
    width=xmax - xmin,
    height=ymax - ymin,
    fill=False,
    linestyle="--",
    linewidth=1.5 * SCALE,
    edgecolor="black",
    zorder=1,
)
ax.add_patch(bbox)

# Create edges: Input -> Latent and Latent -> Output
# Input to Latent edges (skip last input token: i < N_TOKENS - 1)
edges_input_latent = [[None for _ in range(N_LATENT)] for _ in range(N_TOKENS)]
for i in range(N_TOKENS):
    for j in range(N_LATENT):
        if i < N_TOKENS - 1:  # Skip edges from last input token
            (line,) = ax.plot(
                [x_positions[i], x_latent_positions[j]],
                [y_input, y_latent],
                color=EDGE_COLOR,
                alpha=EDGE_ALPHA_INACTIVE,
                linewidth=1.5 * SCALE,
                zorder=2,
            )
            edges_input_latent[i][j] = line

# Latent to Output edges (skip first output token: j > 0)
edges_latent_output = [[None for _ in range(N_TOKENS)] for _ in range(N_LATENT)]
for i in range(N_LATENT):
    for j in range(N_TOKENS):
        if j > 0:  # Skip edges to first output token
            (line,) = ax.plot(
                [x_latent_positions[i], x_positions[j]],
                [y_latent, y_output],
                color=EDGE_COLOR,
                alpha=EDGE_ALPHA_INACTIVE,
                linewidth=1.5 * SCALE,
                zorder=2,
            )
            edges_latent_output[i][j] = line

# Clean up axes
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------- animation logic ---------------------- #
def set_input_style(idx, active: bool):
    patch = input_patches[idx]
    if patch is not None:  # Skip if token doesn't exist
        if active:
            patch.set_facecolor(INPUT_ACTIVE_COLOR)
        else:
            patch.set_facecolor(INPUT_INACTIVE_COLOR)


def set_latent_style(idx, active: bool):
    patch = latent_patches[idx]
    if active:
        patch.set_facecolor(LATENT_ACTIVE_COLOR)
    else:
        patch.set_facecolor(LATENT_INACTIVE_COLOR)


def set_output_style(idx, active: bool):
    patch = output_patches[idx]
    if patch is not None:  # Skip if token doesn't exist
        if active:
            patch.set_facecolor(OUTPUT_ACTIVE_COLOR)
        else:
            patch.set_facecolor(OUTPUT_INACTIVE_COLOR)


def update(frame):
    """
    frame k:
      - active inputs:  0..k
      - active outputs: 0..min(k+1, N_TOKENS-1)
      - connect all latents to inputs [0:k]
      - connect all latents to outputs [0:k+1]
    """
    k = frame
    max_output = min(k + 1, N_TOKENS - 1)
    has_active_input = k >= 0
    
    # update token colors
    for i in range(N_TOKENS):
        set_input_style(i, active=(i <= k))
        set_output_style(i, active=(i <= max_output))
    
    # Latent nodes are active when there are active inputs
    for i in range(N_LATENT):
        set_latent_style(i, active=has_active_input)

    # update edges: Input -> Latent
    # Connect all latents to inputs [0:k]
    for i in range(N_TOKENS):
        for j in range(N_LATENT):
            if edges_input_latent[i][j] is not None:  # Only update edges that exist
                active_edge = (i <= k)
                edges_input_latent[i][j].set_alpha(
                    EDGE_ALPHA_ACTIVE if active_edge else EDGE_ALPHA_INACTIVE
                )
    
    # update edges: Latent -> Output
    # Connect all latents to outputs [0:k+1]
    for i in range(N_LATENT):
        for j in range(N_TOKENS):
            if edges_latent_output[i][j] is not None:  # Only update edges that exist
                active_edge = (j <= max_output)
                edges_latent_output[i][j].set_alpha(
                    EDGE_ALPHA_ACTIVE if active_edge else EDGE_ALPHA_INACTIVE
                )

    return []  # not using blitting; return value not important


# total frames = N_TOKENS (k = 0..7)
anim = FuncAnimation(
    fig,
    update,
    frames=N_TOKENS - 1,
    interval=INTERVAL,
    repeat=True,
)

# ---------------------- save as GIF ---------------------- #
print("Saving high-resolution GIF... this may take a few seconds.")
writer = PillowWriter(fps=FPS)
anim.save("flare_decoder.gif", writer=writer, dpi=DPI)
print("Saved as flare_decoder.gif")

# If you want an interactive preview instead of auto-saving, comment out the
# 'anim.save...' lines above and uncomment:
# plt.show()
