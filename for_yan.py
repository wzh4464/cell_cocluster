# -*- coding: utf-8 -*-
"""
Demo plotting for morphogenesis analysis figures
- English-only figures
- All font sizes and key params defined as constants at the top
- Pure matplotlib + numpy (no external dependencies)
- Saves figures into ./figs
- Supports font scaling via command-line argument
- Supports Times New Roman font via command-line option

Usage:
    python for_yan.py                          # Default settings
    python for_yan.py --font-scale 1.5         # 1.5x larger fonts
    python for_yan.py --use-times               # Use Times New Roman
    python for_yan.py --font-scale 1.2 --use-times  # Both options
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Font scaling configuration
# =========================
class FontConfig:
    """Font configuration class with scaling support"""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        
    @property
    def suptitle_size(self) -> int:
        return int(20 * self.scale_factor)
    
    @property
    def title_size(self) -> int:
        return int(18 * self.scale_factor)
    
    @property
    def label_size(self) -> int:
        return int(14 * self.scale_factor)
    
    @property
    def tick_size(self) -> int:
        return int(12 * self.scale_factor)
    
    @property
    def legend_size(self) -> int:
        return int(12 * self.scale_factor)
    
    @property
    def annot_size(self) -> int:
        return int(10 * self.scale_factor)

# Global font config (will be set by main)
FONT_CONFIG = FontConfig()

LINEWIDTH   = 2.0
MARKERSIZE  = 4
ALPHA_FILL  = 0.22
DPI         = 300

CMAP_HEAT   = "viridis"
CMAP_VZ     = "coolwarm"  # for signed vz
SEED        = 42

COLOR_LEFT  = "#1f77b4"   # blue
COLOR_RIGHT = "#d62728"   # red
COLOR_OURS  = "#2ca02c"   # green
COLOR_THR   = "#ff7f0e"   # orange
COLOR_CORR  = "#9467bd"   # purple
COLOR_DTW   = "#8c564b"   # brown
COLOR_GRAY  = "#7f7f7f"

SAVE_DIR    = "figs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# Optional: set some rcParams defaults (we still explicitly set fonts where needed)
plt.rcParams["figure.dpi"] = DPI
plt.rcParams["savefig.dpi"] = DPI
plt.rcParams["axes.unicode_minus"] = True


# =========================
# Demo data generators
# =========================
def gen_pairwise_coclust_matrix(n_cells=10):
    """
    Generate a symmetric pairwise co-clustering probability matrix with block structure.
    Returns (P, labels)
    """
    labels = [f"hyp{i+1}L" for i in range(n_cells)]
    P = np.zeros((n_cells, n_cells), dtype=float)

    # Two sub-blocks for illustrative structure
    block_size = n_cells // 2
    base_cross = 0.35
    base_within = 0.85

    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                P[i, j] = 1.0
            else:
                same_block = (i < block_size and j < block_size) or (i >= block_size and j >= block_size)
                mean_val = base_within if same_block else base_cross
                P[i, j] = mean_val + rng.normal(0.0, 0.05)
    P = np.clip((P + P.T) / 2.0, 0.0, 1.0)
    return P, labels


def gen_trajectories(n_cells=10, n_time=36, t_min=220, t_max=255):
    """
    Generate synthetic left/right cohort trajectories approaching the midline (y=0).
    Returns times, left_tracks, right_tracks with shape (n_cells, n_time, 2) for (x,y).
    """
    times = np.linspace(t_min, t_max, n_time)

    # Logistic progress from 220 to 245 (center ~235)
    center = 235.0
    k = 0.5
    progress = 1.0 / (1.0 + np.exp(-k * (times - center)))  # 0..1

    # Left cohort starts y>0, moves toward 0; Right cohort starts y<0, moves toward 0
    left_tracks = np.zeros((n_cells, n_time, 2))
    right_tracks = np.zeros((n_cells, n_time, 2))

    # AP drift small sinusoidal + noise
    ap_drift = 0.10 * np.sin((times - t_min) / (t_max - t_min) * 2 * np.pi)

    for c in range(n_cells):
        # Random initial positions
        y0_left  = 0.85 + rng.normal(0.0, 0.06)
        y0_right = -0.85 + rng.normal(0.0, 0.06)
        x0_left  = rng.normal(0.0, 0.05)
        x0_right = rng.normal(0.0, 0.05)

        # X coordinate: mild AP drift plus small idiosyncratic noise
        left_x  = x0_left  + ap_drift + rng.normal(0.0, 0.01, size=n_time)
        right_x = x0_right + ap_drift + rng.normal(0.0, 0.01, size=n_time)

        # Y coordinate: approach 0 following logistic progress
        left_y  = y0_left  * (1.0 - progress) + rng.normal(0.0, 0.02, size=n_time)
        right_y = y0_right * (1.0 - progress) + rng.normal(0.0, 0.02, size=n_time)

        left_tracks[c, :, 0]  = left_x
        left_tracks[c, :, 1]  = left_y
        right_tracks[c, :, 0] = right_x
        right_tracks[c, :, 1] = right_y

    return times, left_tracks, right_tracks


def gen_velocity_field(n_points=80, radius=1.0):
    """
    Generate a toy intestinal velocity field:
    - positions in AP-LR plane
    - in-plane velocities small and inward to center
    - vz negative (ventral/inward) with a Gaussian profile
    Returns X, Y, VX, VY, VZ
    """
    # Random points within circle
    angles = rng.uniform(0, 2 * np.pi, size=n_points)
    radii = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
    X = radii * np.cos(angles)
    Y = radii * np.sin(angles)

    # In-plane velocities point inward
    # magnitude small and proportional to distance from center + small noise
    VX = -0.15 * X + rng.normal(0.0, 0.02, size=n_points)
    VY = -0.15 * Y + rng.normal(0.0, 0.02, size=n_points)

    # Out-of-plane velocity: negative, strongest near center
    r2 = X**2 + Y**2
    VZ = -0.45 * np.exp(-r2 / (2 * (0.5**2))) + rng.normal(0.0, 0.03, size=n_points)  # ~[-0.5, 0]
    return X, Y, VX, VY, VZ


def gen_feature_weights():
    """
    Return weights for two pies (DI vs Intestinal). Values sum to 1 for each group.
    """
    # Dorsal intercalation
    labels_A = ["Y-velocity", "Elongation", "Curvature", "Other geometry"]
    weights_A = np.array([0.26, 0.21, 0.18, 0.35], dtype=float)

    # Intestinal morphogenesis
    labels_B = ["Z-velocity", "Apical area", "Volume change", "Other"]
    weights_B = np.array([0.28, 0.24, 0.19, 0.29], dtype=float)
    return (labels_A, weights_A), (labels_B, weights_B)


def gen_baseline_scores():
    """
    Demo alignment score means + std for four methods.
    """
    methods = ["DiMergeTCC", "Thresholding", "Corr+Community", "DTW Template"]
    means   = np.array([0.87, 0.68, 0.56, 0.60])
    stds    = np.array([0.05, 0.02, 0.09, 0.12])
    colors  = [COLOR_OURS, COLOR_THR, COLOR_CORR, COLOR_DTW]
    return methods, means, stds, colors


def gen_Pbar_curves(t_min=220, t_max=255, step=1):
    """
    Demo time-resolved average co-clustering probabilities Pbar(t) for methods.
    Returns times, dict of method -> (mean_curve, ci_halfwidth)
    """
    times = np.arange(t_min, t_max + 1, step)

    def gaussian_bump(t, mu, sigma, base, amp):
        return base + amp * np.exp(-0.5 * ((t - mu) / sigma)**2)

    # Curves centered around 230 with different amplitudes and baselines
    mu, sigma = 230.0, 4.0
    curves = {
        "DiMergeTCC":   (gaussian_bump(times, mu, sigma, base=0.30, amp=0.60), 0.05),
        "Thresholding": (gaussian_bump(times, mu, sigma, base=0.25, amp=0.40), 0.06),
        "Corr+Community": (gaussian_bump(times, mu, sigma, base=0.25, amp=0.35), 0.07),
        "DTW Template": (gaussian_bump(times, mu, sigma, base=0.22, amp=0.33), 0.07),
    }
    return times, curves


# =========================
# Plotters
# =========================
def plot_heatmap_coclust(P, labels, fname):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(P, cmap=CMAP_HEAT, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.set_title("Dorsal Intercalation (Left): Pairwise Co-clustering", fontsize=FONT_CONFIG.title_size)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_CONFIG.tick_size, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=FONT_CONFIG.tick_size)
    ax.set_xlabel("Cells", fontsize=FONT_CONFIG.label_size)
    ax.set_ylabel("Cells", fontsize=FONT_CONFIG.label_size)
    ax.tick_params(axis="both", labelsize=FONT_CONFIG.tick_size)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-clustering probability", fontsize=FONT_CONFIG.label_size)
    cbar.ax.tick_params(labelsize=FONT_CONFIG.tick_size)

    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(times, left_tracks, right_tracks, fname):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    # Plot individual tracks (thin)
    for c in range(left_tracks.shape[0]):
        ax.plot(left_tracks[c, :, 0], left_tracks[c, :, 1],
                color=COLOR_LEFT, alpha=0.5, lw=1.0)
    for c in range(right_tracks.shape[0]):
        ax.plot(right_tracks[c, :, 0], right_tracks[c, :, 1],
                color=COLOR_RIGHT, alpha=0.5, lw=1.0)

    # Plot mean tracks (thick)
    left_mean  = left_tracks.mean(axis=0)
    right_mean = right_tracks.mean(axis=0)
    ax.plot(left_mean[:, 0], left_mean[:, 1], color=COLOR_LEFT, lw=LINEWIDTH,
            label="Left cohort (mean)")
    ax.plot(right_mean[:, 0], right_mean[:, 1], color=COLOR_RIGHT, lw=LINEWIDTH,
            label="Right cohort (mean)")

    ax.axhline(0.0, color=COLOR_GRAY, lw=1.0, ls="--", alpha=0.8)

    ax.set_title("Dorsal Intercalation: Cell Trajectories", fontsize=FONT_CONFIG.title_size)
    ax.set_xlabel("Anterior–Posterior (AP, x)", fontsize=FONT_CONFIG.label_size)
    ax.set_ylabel("Left–Right (LR, y)", fontsize=FONT_CONFIG.label_size)
    ax.tick_params(axis="both", labelsize=FONT_CONFIG.tick_size)
    ax.legend(fontsize=FONT_CONFIG.legend_size, frameon=False)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_field(X, Y, VX, VY, VZ, fname):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))

    # Colored points by vz
    sc = ax.scatter(X, Y, c=VZ, cmap=CMAP_VZ, vmin=-0.5, vmax=0.5, s=30, edgecolor="k", linewidth=0.3)

    # In-plane quiver (downsample if needed)
    ax.quiver(X, Y, VX, VY, color="k", alpha=0.6, width=0.003, scale=10)

    ax.set_title("Intestinal Morphogenesis: Velocity Field", fontsize=FONT_CONFIG.title_size)
    ax.set_xlabel("Anterior–Posterior (AP, x)", fontsize=FONT_CONFIG.label_size)
    ax.set_ylabel("Left–Right (LR, y)", fontsize=FONT_CONFIG.label_size)
    ax.tick_params(axis="both", labelsize=FONT_CONFIG.tick_size)
    ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("vz (DV axis; ventral negative)", fontsize=FONT_CONFIG.label_size)
    cbar.ax.tick_params(labelsize=FONT_CONFIG.tick_size)

    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_feature_pies(fname):
    (labels_A, wA), (labels_B, wB) = gen_feature_weights()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    explode = [0.03] * len(wA)

    wedges1, texts1, autotexts1 = axes[0].pie(
        wA, labels=labels_A, autopct=lambda p: f"{p:.0f}%",
        startangle=90, explode=explode,
        textprops={"fontsize": FONT_CONFIG.tick_size},
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"}
    )
    for t in autotexts1:
        t.set_fontsize(FONT_CONFIG.tick_size)
    axes[0].set_title("Dorsal intercalation: feature weights", fontsize=FONT_CONFIG.title_size)
    axes[0].axis("equal")

    explode2 = [0.03] * len(wB)
    wedges2, texts2, autotexts2 = axes[1].pie(
        wB, labels=labels_B, autopct=lambda p: f"{p:.0f}%",
        startangle=90, explode=explode2,
        textprops={"fontsize": FONT_CONFIG.tick_size},
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"}
    )
    for t in autotexts2:
        t.set_fontsize(FONT_CONFIG.tick_size)
    axes[1].set_title("Intestinal morphogenesis: feature weights", fontsize=FONT_CONFIG.title_size)
    axes[1].axis("equal")

    fig.suptitle("Morphogenetic Feature Distributions", fontsize=FONT_CONFIG.suptitle_size)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# Nature Communications-friendly (Okabe–Ito) palette
NC_PALETTE = {
    "green":      "#009E73",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "yellow":     "#F0E442",
    "black":      "#000000",
    "grey":       "#999999",
}
# Map methods to palette colors (adjust names to your actual method labels)
METHOD_COLOR_MAP = {
    "DiMergeTCC":      NC_PALETTE["green"],
    "Thresholding":    NC_PALETTE["orange"],
    "Corr+Community":  NC_PALETTE["blue"],
    "DTW Template":    NC_PALETTE["purple"],
}
# Bar and errorbar style constants
BAR_WIDTH       = 0.08   # thinner than default 0.8
BAR_EDGE_LW     = 0.6    # thin bar edge line
ERR_ECOLOR      = "#333333"
ERR_ELINEWIDTH  = 1.1
ERR_CAPSIZE     = 2.5
ERR_CAPTHICK    = 1.1
X_SPACING       = 0.2   # compress x-spacing so bars look closer
GRID_ALPHA      = 0.25   # faint horizontal grid for readability

def plot_baseline_bars(fname):
    methods, means, stds, _ = gen_baseline_scores()

    # Colors from Nature Communications-friendly palette
    colors = [METHOD_COLOR_MAP.get(m, NC_PALETTE["grey"]) for m in methods]

    # Compress horizontal spacing so bars are closer to each other
    x = np.arange(len(methods)) * X_SPACING

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    bars = ax.bar(
        x, means,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="#333333",
        linewidth=BAR_EDGE_LW,
        yerr=stds,
        error_kw={
            "elinewidth": ERR_ELINEWIDTH,
            "capthick":   ERR_CAPTHICK,
            "capsize":    ERR_CAPSIZE,
            "ecolor":     ERR_ECOLOR
        },
    )

    # Axis styling: remove top/right spines, add faint horizontal grid under bars
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=GRID_ALPHA, linewidth=0.8)

    # Ticks and labels (English), all fonts from FONT_CONFIG
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=FONT_CONFIG.tick_size, rotation=15, ha="right")
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("Alignment score", fontsize=FONT_CONFIG.label_size)
    ax.tick_params(axis="y", labelsize=FONT_CONFIG.tick_size)

    # Reduce side margins so bars appear closer to figure edges
    ax.margins(x=0.02)

    # Annotate numeric values above bars; place just above the error bar
    for i, (m, s) in enumerate(zip(means, stds)):
        top = m + s
        ax.text(
            x[i], min(0.98, top + 0.03),
            f"{m:.2f} ± {s:.2f}",
            ha="center", va="bottom", fontsize=FONT_CONFIG.annot_size
        )

    # No title per your request; caption should describe metrics and cohorts
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



def plot_Pbar_overlay(fname, theta=0.80, win=(225, 235)):
    times, curves = gen_Pbar_curves()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Shaded expected window
    ax.axvspan(win[0], win[1], color=COLOR_GRAY, alpha=0.15, label="Expected window")

    # Plot each method with CI bands
    for name, color in [("DiMergeTCC", COLOR_OURS),
                        ("Thresholding", COLOR_THR),
                        ("Corr+Community", COLOR_CORR),
                        ("DTW Template", COLOR_DTW)]:
        y, ci = curves[name]
        ax.plot(times, y, color=color, lw=LINEWIDTH, label=name)
        ax.fill_between(times, y - ci, y + ci, color=color, alpha=ALPHA_FILL, linewidth=0)

    # Threshold line (alignment criterion)
    ax.axhline(theta, color="k", lw=1.0, ls="--", alpha=0.8, label=f"Threshold (θ={theta:.2f})")

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (min relative to end of 4-cell stage)", fontsize=FONT_CONFIG.label_size)
    ax.set_ylabel("Mean co-clustering probability, P̄(t)", fontsize=FONT_CONFIG.label_size)
    ax.set_title("Time-resolved coordination: methods overlay", fontsize=FONT_CONFIG.title_size)
    ax.tick_params(axis="both", labelsize=FONT_CONFIG.tick_size)
    ax.legend(fontsize=FONT_CONFIG.legend_size, frameon=False, ncol=2)

    fig.tight_layout()
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main: build all figures
# =========================
def main():
    global FONT_CONFIG
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate morphogenesis demo figures')
    parser.add_argument('--font-scale', type=float, default=1.0,
                       help='Font scaling factor (default: 1.0, e.g. 1.5 means 1.5x larger)')
    parser.add_argument('--use-times', action='store_true',
                       help='Use Times New Roman font family')
    
    args = parser.parse_args()
    
    # Initialize font configuration
    FONT_CONFIG = FontConfig(scale_factor=args.font_scale)
    
    # Set font family if requested
    if args.use_times:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = 'Times New Roman'
    
    print("🔬 Morphogenesis Demo System")
    print("=" * 40)
    print(f"Font scale factor: {args.font_scale}")
    print(f"Using Times New Roman: {args.use_times}")
    print("Generating 6 demo figures...")
    print()

    # 1) Co-clustering heatmap
    P, labels = gen_pairwise_coclust_matrix(n_cells=10)
    plot_heatmap_coclust(P, labels, fname="fig_co_clust_heatmap.png")

    # 2) Trajectories (left/right cohorts)
    times, left_tracks, right_tracks = gen_trajectories(n_cells=10, n_time=36, t_min=220, t_max=255)
    plot_trajectories(times, left_tracks, right_tracks, fname="fig_trajectories.png")

    # 3) Velocity field (intestinal, vz colored)
    X, Y, VX, VY, VZ = gen_velocity_field(n_points=90, radius=1.0)
    plot_velocity_field(X, Y, VX, VY, VZ, fname="fig_velocity_field.png")

    # 4) Feature weight pies
    plot_feature_pies(fname="fig_feature_pies.png")

    # 5) Baseline comparison bars with error bars
    plot_baseline_bars(fname="fig_baseline_bars.png")

    # 6) P̄(t) overlay for multiple methods
    plot_Pbar_overlay(fname="fig_pbar_overlay.png", theta=0.80, win=(225, 235))

    print()
    print("🎉 Demo generation completed!")
    print(f"All demo figures saved in: ./{SAVE_DIR}")
    print()
    print("📏 Font sizes:")
    print(f"   - Titles: {FONT_CONFIG.title_size}pt")
    print(f"   - Labels: {FONT_CONFIG.label_size}pt") 
    print(f"   - Ticks: {FONT_CONFIG.tick_size}pt")
    print(f"   - Legend: {FONT_CONFIG.legend_size}pt")


if __name__ == "__main__":
    main()
