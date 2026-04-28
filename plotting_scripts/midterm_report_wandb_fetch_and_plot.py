"""
Midterm Report — Fetch from WandB & Plot (v6 - two separate plots)
Humanoid Locomotion with Flow Matching — 16-831 IRL Project
Simson D'Souza & Rohit Satishkumar, CMU

Run:
    python fetch_and_plot_v6.py

Outputs:
    midterm_walk.png   — Walk task plot
    midterm_stair.png  — Stair task plot
"""

import pathlib

import wandb
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── YOUR RUN IDs ─────────────────────────────────────────────────────────────
ENTITY  = "intro_to_robot_learning_cmu"
PROJECT = "humanoid-bench"

RUNS = {
    "SAC_stair":    "sac_g1-stair-v0_20260319_214847",
    "SAC_walk":     "sac_g1-walk-v0_20260319_041427",
    "PPO_stair":    "qa65deg0",
    "PPO_walk":     "bo3xdqvp",
    "TDMPC2_stair": "xvlqq8yi",
    "TDMPC2_walk":  "6z7xruhf",
    "FPO_stair":    "0np8q4p6",
    "FPO_walk":     "1hsgb1ay",
}

# ─── CONFIRMED METRIC NAMES ───────────────────────────────────────────────────
SAC_METRIC = "evaluation/episode.return"
SAC_STEP   = "_step"        # = training step i, goes to 10M

PPO_METRIC = "results/return"
PPO_STEP   = "global_step"  # = actual env steps, goes to 20M ✅

TDMPC2_METRIC = "results/return"
TDMPC2_STEP   = "_step"     # = actual env steps (same as train/step)

FPO_METRIC = "results/return"
FPO_STEP   = "global_step"  # actual env steps logged as a data column (step=i is just the outer iter index)

# ─── RANDOM AGENT (from your proposal Table 1) ────────────────────────────────
RANDOM_WALK  = 2.554
RANDOM_STAIR = 1.452

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

COLORS = {
    "Random": "#888888",
    "PPO":    "#E76F51",
    "SAC":    "#2A9D8F",
    "TDMPC2": "#9B59B6",
    "FPO":    "#F4A261",
}


def smooth(y, window_frac=0.05):
    w = max(3, int(len(y) * window_frac))
    kernel = np.ones(w) / w
    padded = np.pad(y, (w//2, w//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(y)]


def fetch_run(run_id, metric, step_col):
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}/runs/{run_id}"
    print(f"  Fetching: {run_id}")
    run = api.run(path)
    history = run.history(samples=10000, pandas=True)

    if step_col not in history.columns:
        print(f"  WARNING: '{step_col}' not found, falling back to _step")
        step_col = "_step"

    if metric not in history.columns:
        print(f"  ERROR: '{metric}' not found!")
        return None, None

    df = history[[step_col, metric]].dropna()
    if df.empty:
        print(f"  WARNING: empty after dropna")
        return None, None

    steps   = df[step_col].values.astype(float)
    returns = df[metric].values.astype(float)

    steps_millions = steps / 1e6

    print(f"  Datapoints : {len(steps)}")
    print(f"  Steps      : {steps_millions[0]:.3f}M → {steps_millions[-1]:.3f}M")
    print(f"  Return     : {returns.min():.2f} → {returns.max():.2f}")

    returns_smooth = smooth(returns, window_frac=0.05)
    return steps_millions, returns_smooth


def save_single_plot(filename, title, random_mean,
                     ppo_steps, ppo_ret,
                     sac_steps, sac_ret,
                     tdmpc2_steps=None, tdmpc2_ret=None,
                     fpo_steps=None, fpo_ret=None,
                     y_max=None, legend_loc="upper left"):
    """Save one standalone figure for a single task."""

    fig, ax = plt.subplots(figsize=(8, 5))

    # x limit
    x_max = 0
    if sac_steps is not None:
        x_max = max(x_max, sac_steps.max())
    if ppo_steps is not None:
        x_max = max(x_max, ppo_steps.max())
    if tdmpc2_steps is not None:
        x_max = max(x_max, tdmpc2_steps.max())
    if fpo_steps is not None:
        x_max = max(x_max, fpo_steps.max())
    x_max = max(x_max, 1.0)

    # Random baseline
    ax.axhline(random_mean, color=COLORS["Random"], linestyle="--",
               linewidth=1.8, zorder=2,
               label=f"Random Agent (mean = {random_mean:.3f})")

    # PPO
    if ppo_steps is not None:
        ax.plot(ppo_steps, ppo_ret, color=COLORS["PPO"],
                linewidth=2.2, label="PPO (On-policy)", zorder=4)

    # SAC
    if sac_steps is not None:
        ax.plot(sac_steps, sac_ret, color=COLORS["SAC"],
                linewidth=2.2, label="SAC (Off-policy)", zorder=3)

    # TDMPC2
    if tdmpc2_steps is not None:
        ax.plot(tdmpc2_steps, tdmpc2_ret, color=COLORS["TDMPC2"],
                linewidth=2.2, label="TDMPC2 (Model-based)", zorder=5)

    # FPO
    if fpo_steps is not None:
        ax.plot(fpo_steps, fpo_ret, color=COLORS["FPO"],
                linewidth=2.2, label="FPO (Modification 1)", zorder=6)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Environment Steps (millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_xlim(0, x_max * 1.02)
    if y_max is not None:
        ax.set_ylim(bottom=None, top=y_max)
    ax.legend(loc=legend_loc, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def main():
    print("=== Fetching WandB runs ===\n")

    print("PPO Walk:")
    ppo_walk_steps,  ppo_walk_ret  = fetch_run(
        RUNS["PPO_walk"],  PPO_METRIC, PPO_STEP)

    print("\nPPO Stair:")
    ppo_stair_steps, ppo_stair_ret = fetch_run(
        RUNS["PPO_stair"], PPO_METRIC, PPO_STEP)

    print("\nSAC Walk:")
    sac_walk_steps,  sac_walk_ret  = fetch_run(
        RUNS["SAC_walk"],  SAC_METRIC, SAC_STEP)

    print("\nSAC Stair:")
    sac_stair_steps, sac_stair_ret = fetch_run(
        RUNS["SAC_stair"], SAC_METRIC, SAC_STEP)

    print("\nTDMPC2 Walk:")
    tdmpc2_walk_steps, tdmpc2_walk_ret = fetch_run(
        RUNS["TDMPC2_walk"], TDMPC2_METRIC, TDMPC2_STEP)

    print("\nTDMPC2 Stair:")
    tdmpc2_stair_steps, tdmpc2_stair_ret = fetch_run(
        RUNS["TDMPC2_stair"], TDMPC2_METRIC, TDMPC2_STEP)

    print("\nFPO Walk:")
    fpo_walk_steps, fpo_walk_ret = fetch_run(
        RUNS["FPO_walk"], FPO_METRIC, FPO_STEP)

    print("\nFPO Stair:")
    fpo_stair_steps, fpo_stair_ret = fetch_run(
        RUNS["FPO_stair"], FPO_METRIC, FPO_STEP)

    print("\n=== Saving separate plots ===")

    save_single_plot(
        filename=OUTPUT_DIR / "walk.png",
        title="Walk Task  (g1-walk-v0)",
        random_mean=RANDOM_WALK,
        ppo_steps=ppo_walk_steps,   ppo_ret=ppo_walk_ret,
        sac_steps=sac_walk_steps,   sac_ret=sac_walk_ret,
        tdmpc2_steps=tdmpc2_walk_steps, tdmpc2_ret=tdmpc2_walk_ret,
        fpo_steps=fpo_walk_steps,       fpo_ret=fpo_walk_ret,
        legend_loc="upper right",
    )

    save_single_plot(
        filename=OUTPUT_DIR / "stair.png",
        title="Stair Task  (g1-stair-v0)",
        random_mean=RANDOM_STAIR,
        ppo_steps=ppo_stair_steps,  ppo_ret=ppo_stair_ret,
        sac_steps=sac_stair_steps,  sac_ret=sac_stair_ret,
        tdmpc2_steps=tdmpc2_stair_steps, tdmpc2_ret=tdmpc2_stair_ret,
        fpo_steps=fpo_stair_steps,       fpo_ret=fpo_stair_ret,
        legend_loc="upper right",
    )

    save_single_plot(
        filename=OUTPUT_DIR / "walk_no_tdmpc2.png",
        title="Walk Task  (g1-walk-v0)",
        random_mean=RANDOM_WALK,
        ppo_steps=ppo_walk_steps,   ppo_ret=ppo_walk_ret,
        sac_steps=sac_walk_steps,   sac_ret=sac_walk_ret,
        fpo_steps=fpo_walk_steps,   fpo_ret=fpo_walk_ret,
    )

    save_single_plot(
        filename=OUTPUT_DIR / "stair_no_tdmpc2.png",
        title="Stair Task  (g1-stair-v0)",
        random_mean=RANDOM_STAIR,
        ppo_steps=ppo_stair_steps,  ppo_ret=ppo_stair_ret,
        sac_steps=sac_stair_steps,  sac_ret=sac_stair_ret,
        fpo_steps=fpo_stair_steps,  fpo_ret=fpo_stair_ret,
    )

    print("\nDone! Four files saved:")
    print("walk.png")
    print("stair.png")
    print("walk_no_tdmpc2.png")
    print("stair_no_tdmpc2.png")


if __name__ == "__main__":
    main()