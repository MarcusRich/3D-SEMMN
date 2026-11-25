"""Draw the README figures into docs/figures/.

Every figure is simplified on purpose, the way a textbook picture is. Each one is badged
Measured (drawn from this repository's results), Planned (the roadmap's brain mapping) or
Illustration (a concept sketch), and the README says what each simplifies.

    python make_figures.py                    # all figures
    python make_figures.py --only brain wiring

Needs requirements-figures.txt. The brain atlas (Schaefer 2018, 100 parcels, 7 networks) and
the fsaverage5 surface come from nilearn's public downloads and are never committed. The word
vectors in the meaning-map figure come from the local LM cache next to the Broderick data
($SEMMN_BRODERICK/cache/lm_local.npz); that figure is skipped when the cache is absent.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent


def _load_montages():
    """Load semmn/eeg/montages.py by path: the semmn package imports torch, which figures don't need."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("semmn_montages", ROOT / "semmn" / "eeg" / "montages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BESA_1020, mod.STARTER_MONTAGES


BESA_1020, STARTER_MONTAGES = _load_montages()
OUT = ROOT / "docs" / "figures"
EEG = ROOT / "outputs" / "eeg"
SEED = 7
DPI = 160

# Schaefer 2018 7-network colours (the atlas's own lookup table) and plain names.
NETWORKS = {
    "Vis": ("Visual", "#781286"),
    "SomMot": ("Body and hearing", "#4682B4"),
    "DorsAttn": ("Attention (where)", "#00760E"),
    "SalVentAttn": ("Attention (what matters)", "#C43AFA"),
    "Limbic": ("Emotion and memory", "#B8C77A"),
    "Cont": ("Control and planning", "#E69422"),
    "Default": ("Language and inner thought", "#CD3E4E"),
}
INK = "#1f2328"
MUTED = "#6e7781"
PANEL = "#ffffff"
BADGE = {"Measured": "#1a7f37", "Planned": "#0969da", "Illustration": "#8250df"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": PANEL,
        "savefig.facecolor": PANEL,
    }
)


# --------------------------------------------------------------------------- helpers


def badge(fig, kind: str) -> None:
    """Mark the figure Measured, Planned or Illustration; save() pins it to the drawing's top-right corner."""
    fig._semmn_badge = kind


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    kind = getattr(fig, "_semmn_badge", None)
    if kind:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())  # inches, content only
        w, h = fig.get_size_inches()
        fig.text(bb.x1 / w, bb.y1 / h + 0.012, f"  {kind.upper()}  ", ha="right", va="bottom", fontsize=9,
                 fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.35", fc=BADGE[kind], ec="none"))
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def load_atlas():
    """Schaefer-100 (7 networks): image, network key per parcel, MNI centroid per parcel."""
    from nilearn import datasets, plotting

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0)
    labels = [lb.decode() if isinstance(lb, bytes) else str(lb) for lb in atlas.labels]
    labels = [lb for lb in labels if lb.lower() != "background"]
    nets = [lb.split("_")[2] for lb in labels]  # 7Networks_LH_Vis_1 -> Vis
    hemis = [lb.split("_")[1] for lb in labels]
    coords = plotting.find_parcellation_cut_coords(atlas.maps)
    return atlas.maps, labels, nets, hemis, np.asarray(coords)


def electrode_xyz(name: str) -> np.ndarray:
    """Unit-sphere position from the BESA (theta, phi) convention in semmn/eeg/montages.py."""
    theta, phi = np.deg2rad(BESA_1020[name])
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def electrode_mni(name: str) -> np.ndarray:
    """A rough template-head placement: the unit sphere scaled into MNI millimetres.

    Illustration only. A real mapping co-registers measured positions to a head model.
    """
    x, y, z = electrode_xyz(name)
    return np.array([70 * x, 88 * y - 18, 72 * z + 22])


def scalp_xy(name: str) -> np.ndarray:
    """Top-down azimuthal projection (nose up), radius 1 at the 10-20 equator."""
    x, y, _ = electrode_xyz(name)
    theta = abs(BESA_1020[name][0])
    direction = np.array([x, y]) / max(np.hypot(x, y), 1e-9)
    return direction * theta / 92 * 0.92


# --------------------------------------------------------------------------- figures


def fig_big_idea() -> None:
    fig = plt.figure(figsize=(14, 3.9))
    badge(fig, "Illustration")
    ax = fig.add_axes([0.01, 0.0, 0.98, 0.84])
    ax.set_xlim(0, 14)
    ax.set_ylim(0.1, 3.25)
    ax.axis("off")
    rng = np.random.default_rng(SEED)

    steps = [
        ("EEG cap", "8-16 cheap\nelectrodes"),
        ("Brain waves", "tiny voltages,\nmostly noise"),
        ("Brain regions", "which networks\nare busy"),
        ("Model columns", "100 small units,\none per region"),
        ("Meaning map", "a few numbers\nthat capture gist"),
        ("Word guesses", "the language model\nranks candidates"),
        ("You choose", "nothing is said\nwithout a yes"),
    ]
    xs = np.linspace(1.0, 13.0, len(steps))
    for i, (x, (head, sub)) in enumerate(zip(xs, steps)):
        ax.add_patch(FancyBboxPatch((x - 0.8, 1.15), 1.6, 1.95, boxstyle="round,pad=0.05,rounding_size=0.25",
                                    fc="#f6f8fa", ec="#d0d7de", lw=1.2))
        cx, cy = x, 2.13
        if i == 0:  # head with electrodes
            ax.add_patch(Circle((cx, cy), 0.48, fc="#ffe8d6", ec=INK, lw=1.2))
            ax.add_patch(Polygon([[cx - 0.08, cy + 0.47], [cx, cy + 0.6], [cx + 0.08, cy + 0.47]], fc="#ffe8d6", ec=INK))
            for name in STARTER_MONTAGES["cyton8"]:
                px, py = scalp_xy(name) * 0.46
                ax.add_patch(Circle((cx + px, cy + py), 0.06, fc="#0969da", ec="white", lw=0.6))
        elif i == 1:  # wiggly traces
            t = np.linspace(-0.6, 0.6, 200)
            for k in range(4):
                sig = 0.05 * np.sin(18 * t + k) + 0.035 * rng.standard_normal(t.size)
                ax.plot(cx + t, cy + 0.36 - 0.24 * k + sig, color=["#0969da", "#8250df", "#cf222e", "#1a7f37"][k], lw=1)
        elif i == 2:  # brain blob with coloured patches
            th = np.linspace(0, 2 * np.pi, 100)
            r = 0.5 + 0.05 * np.sin(5 * th)
            ax.fill(cx + r * np.cos(th) * 1.1, cy + r * np.sin(th) * 0.8, color="#f2d7dd", ec=INK, lw=1)
            for (px, py), col in zip([(-0.25, 0.1), (0.2, 0.15), (0.0, -0.2), (0.3, -0.15)],
                                     [NETWORKS[k][1] for k in ("Vis", "Default", "SomMot", "Cont")]):
                ax.add_patch(Circle((cx + px, cy + py), 0.13, fc=col, alpha=0.85))
        elif i == 3:  # grid of columns
            cols = [v[1] for v in NETWORKS.values()]
            for a in range(5):
                for b in range(5):
                    ax.add_patch(Circle((cx - 0.48 + a * 0.24, cy - 0.48 + b * 0.24), 0.085,
                                        fc=cols[(a * 5 + b) % 7], alpha=0.9))
        elif i == 4:  # curved sheet with dots
            u = np.linspace(-0.6, 0.6, 30)
            ax.plot(cx + u, cy + 0.3 * np.sin(3 * u) + 0.15, color=MUTED, lw=1)
            ax.plot(cx + u, cy + 0.3 * np.sin(3 * u) - 0.15, color=MUTED, lw=1)
            pts = rng.uniform(-0.55, 0.55, 14)
            ax.scatter(cx + pts, cy + 0.3 * np.sin(3 * pts) + rng.uniform(-0.12, 0.12, 14),
                       s=22, c=rng.choice(["#0969da", "#cf222e", "#1a7f37"], 14), zorder=3)
        elif i == 5:  # ranked words
            for k, (w, p) in enumerate([("water", 0.46), ("outside", 0.27), ("tired", 0.17)]):
                ax.add_patch(FancyBboxPatch((cx - 0.62, cy + 0.32 - k * 0.34), 1.24 * p / 0.46, 0.24,
                                            boxstyle="round,pad=0.01", fc="#ddf4ff", ec="#54aeff"))
                ax.text(cx - 0.56, cy + 0.44 - k * 0.34, w, fontsize=9, va="center")
        else:  # check mark and speech
            ax.add_patch(Circle((cx, cy), 0.45, fc="#dafbe1", ec="#1a7f37", lw=1.5))
            ax.plot([cx - 0.2, cx - 0.04, cx + 0.24], [cy, cy - 0.17, cy + 0.2], color="#1a7f37", lw=3)
        ax.text(x, 0.95, head, ha="center", va="top", fontsize=11.5, fontweight="bold")
        ax.text(x, 0.55, sub, ha="center", va="top", fontsize=9, color=MUTED)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.84, 2.1), (xs[i + 1] - 0.84, 2.1), arrowstyle="-|>",
                                         mutation_scale=14, color=MUTED, lw=1.3))
    save(fig, "01_core_process")


def _majority_smooth(labels: np.ndarray, faces: np.ndarray, n_labels: int, rounds: int) -> np.ndarray:
    """Each vertex takes the most common label among itself and its mesh neighbours.

    Sampling a volume atlas onto a surface leaves speckle at parcel edges; this cleans it up the way a
    textbook atlas figure looks. Label 0 (medial wall) is kept where it is.
    """
    from scipy.sparse import coo_matrix, identity

    n = len(labels)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = (coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr() > 0).astype(float) + identity(n)
    out = labels.copy()
    for _ in range(rounds):
        onehot = np.zeros((n, n_labels))
        onehot[np.arange(n), out] = 1
        votes = adj @ onehot
        votes[:, 0] = 0
        new = votes.argmax(1)
        out = np.where(labels == 0, 0, new)
    return out


def fig_brain(atlas) -> None:
    from nilearn import datasets, plotting, surface

    maps, labels, nets, hemis, _ = atlas
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    keys = list(NETWORKS)
    net_index = np.array([keys.index(n) + 1 for n in nets])  # parcel -> 1..7
    cmap = ListedColormap([NETWORKS[k][1] for k in keys])

    fig = plt.figure(figsize=(13, 7.4))
    badge(fig, "Planned")
    views = [("left", "lateral", "Left side"), ("right", "lateral", "Right side"),
             ("left", "medial", "Left, inner face"), ("right", "medial", "Right, inner face")]
    for k, (hemi, view, name) in enumerate(views):
        mesh = fs[f"pial_{hemi}"]
        parcel = surface.vol_to_surf(maps, mesh, interpolation="nearest", radius=3.0)
        parcel = np.rint(parcel).astype(int)
        roi = np.where(parcel > 0, net_index[np.clip(parcel - 1, 0, 99)], 0)
        roi = _majority_smooth(roi, surface.load_surf_mesh(mesh).faces, n_labels=8, rounds=6).astype(float)
        roi[roi == 0] = np.nan
        ax = fig.add_axes([0.02 + (k % 2) * 0.36, 0.47 - (k // 2) * 0.42, 0.36, 0.42], projection="3d")
        plotting.plot_surf_roi(fs[f"infl_{hemi}"], roi_map=roi, hemi=hemi, view=view,
                               bg_map=fs[f"sulc_{hemi}"], bg_on_data=True, darkness=0.45,
                               cmap=cmap, vmin=1, vmax=7, colorbar=False, axes=ax, figure=fig)
        ax.set_title(name, fontsize=11, color=MUTED, y=0.02)
    lx = 0.755
    fig.text(lx, 0.83, "Networks", fontsize=12, fontweight="bold")
    counts = {k: nets.count(k) for k in keys}
    for i, k in enumerate(keys):
        y = 0.78 - i * 0.062
        fig.patches.append(FancyBboxPatch((lx, y - 0.012), 0.022, 0.03, boxstyle="round,pad=0.002",
                                          transform=fig.transFigure, fc=NETWORKS[k][1], ec="none"))
        fig.text(lx + 0.03, y + 0.003, f"{NETWORKS[k][0]}", fontsize=10.5, va="center")
        fig.text(lx + 0.03, y - 0.022, f"{counts[k]} columns", fontsize=8.5, color=MUTED, va="center")
    fig.text(lx, 0.30, "Today's toy model, mapped later:", fontsize=10, fontweight="bold")
    for i, (a, b) in enumerate([("vision area", "Visual"), ("audio area", "Body and hearing"),
                                ("hub", "Language + Control")]):
        fig.text(lx, 0.26 - i * 0.035, f"{a}  →  {b}", fontsize=9.5, color=MUTED)
    save(fig, "02_columns_on_brain")


def fig_wiring(atlas) -> None:
    from nilearn import plotting

    _, _, nets, _, coords = atlas
    rng = np.random.default_rng(SEED)
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    lam = 22.0  # mm; the model's wiring rule: closer columns connect more often
    p = np.exp(-d / lam)
    np.fill_diagonal(p, 0)
    iu = np.triu_indices(len(coords), 1)
    w = p[iu] / p[iu].sum()
    pick = rng.choice(len(w), size=170, replace=False, p=w)
    adj = np.zeros_like(d)
    adj[iu[0][pick], iu[1][pick]] = 1
    adj = adj + adj.T
    node_colors = [NETWORKS[n][1] for n in nets]

    fig = plt.figure(figsize=(13, 5.6))
    badge(fig, "Planned")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    plotting.plot_connectome(adj, coords, node_color=node_colors, node_size=34, display_mode="lzr",
                             edge_kwargs={"color": "#57606a", "linewidth": 0.7, "alpha": 0.45},
                             node_kwargs={"edgecolors": "white", "linewidths": 0.6}, axes=ax, figure=fig,
                             annotate=False, black_bg=False, colorbar=False)
    save(fig, "03_column_wiring")


def fig_electrodes(atlas) -> None:
    from nilearn import plotting

    _, _, nets, _, coords = atlas
    montage = STARTER_MONTAGES["cyton_daisy16"]
    elec = np.array([electrode_mni(n) for n in montage])
    all_nodes = np.vstack([coords, elec])
    n_par = len(coords)
    adj = np.zeros((len(all_nodes),) * 2)
    for i, e in enumerate(elec):
        dist = np.linalg.norm(coords - e, axis=1)
        for j in np.argsort(dist)[:3]:
            adj[n_par + i, j] = adj[j, n_par + i] = 1.0 / (1 + dist[j] / 20)
    colors = [NETWORKS[n][1] for n in nets] + ["#0a0a0a"] * len(elec)
    sizes = [26] * n_par + [70] * len(elec)

    fig = plt.figure(figsize=(13, 5.6))
    badge(fig, "Illustration")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    plotting.plot_connectome(adj, all_nodes, node_color=colors, node_size=sizes, display_mode="lzr",
                             edge_cmap="Greys", edge_vmin=0, edge_vmax=0.8,
                             edge_kwargs={"linewidth": 1.4, "alpha": 0.8},
                             node_kwargs={"edgecolors": "white", "linewidths": 0.6}, axes=ax, figure=fig,
                             annotate=False, black_bg=False, colorbar=False)
    save(fig, "04_electrodes_to_columns")


def _scalp_regions(tier: str) -> dict[str, float]:
    text = (EEG / "scalp_forward.md").read_text()
    block = text.split(f"## {tier}", 1)[1].split("\n## ", 1)[0]
    return {m.group(1): float(m.group(3)) for m in re.finditer(r"^\| ([a-z]+-[A-Za-z]+) \| (\d+) \| ([\d.]+) \|", block, re.M)}


def _region_of(name: str) -> str:
    side = "mid" if name.endswith("z") else ("L" if int(re.sub(r"\D", "", name) or 0) % 2 else "R")
    stem = re.sub(r"[\dz]", "", name)
    lobe = {"Fp": "frontal", "F": "frontal", "T": "temporal", "C": "central", "P": "parietal",
            "PO": "occipital", "O": "occipital"}[stem]
    return f"{lobe}-{side}"


def fig_headset() -> None:
    from scipy.interpolate import Rbf

    vals = _scalp_regions("C_env_onset_lm")
    names = list(BESA_1020)
    pts = np.array([scalp_xy(n) for n in names])
    z = np.array([vals.get(_region_of(n), np.nan) for n in names])
    ok = ~np.isnan(z)
    rbf = Rbf(pts[ok, 0], pts[ok, 1], z[ok], function="multiquadric", smooth=0.002)
    g = np.linspace(-1.05, 1.05, 260)
    gx, gy = np.meshgrid(g, g)
    gz = rbf(gx, gy)
    gz[np.hypot(gx, gy) > 1.0] = np.nan

    fig = plt.figure(figsize=(13, 5.8))
    badge(fig, "Measured")
    for k, (mname, label) in enumerate([("biosemi128", "Lab cap, 128 electrodes"),
                                         ("cyton_daisy16", "Starter headset, 16"),
                                         ("cyton8", "Starter headset, 8")]):
        ax = fig.add_axes([0.03 + k * 0.31, 0.06, 0.28, 0.72])
        ax.set_aspect("equal")
        ax.axis("off")
        im = ax.contourf(gx, gy, gz, levels=14, cmap="YlOrRd", vmin=0.026, vmax=0.041)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=INK, lw=1.6))
        ax.add_patch(Polygon([[-0.1, 0.99], [0, 1.13], [0.1, 0.99]], fill=False, ec=INK, lw=1.6))
        for sx in (-1, 1):
            ax.add_patch(FancyBboxPatch((sx * 1.0 - 0.05, -0.18), 0.1, 0.36, boxstyle="round,pad=0.02",
                                        fill=False, ec=INK, lw=1.4))
        if mname == "biosemi128":
            # BioSemi's cap is radial-equidistant: rings of electrodes around the vertex (drawn schematically).
            ax.add_patch(Circle((0, 0), 0.03, fc="#0a0a0a", alpha=0.8))
            for ring, rr in enumerate(np.linspace(0.13, 0.95, 7)):
                k = 6 * (ring + 1)
                for a in np.linspace(0, 2 * np.pi, k, endpoint=False) + ring * 0.3:
                    ax.add_patch(Circle((rr * np.cos(a), rr * np.sin(a)), 0.03, fc="#0a0a0a", alpha=0.8))
        else:
            on = set(STARTER_MONTAGES[mname])
            for n, (px, py) in zip(names, pts):
                if n in on:
                    ax.add_patch(Circle((px, py), 0.06, fc="#0a0a0a", ec="white", lw=1.2))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.15, 1.2)
        ax.set_title(label, fontsize=12, pad=4)
    cax = fig.add_axes([0.955, 0.18, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([])
    cax.text(0.5, 1.03, "closer", transform=cax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    cax.text(0.5, -0.06, "weaker", transform=cax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    save(fig, "05_headset_map")


def fig_n400() -> None:
    rows = list(csv.DictReader((EEG / "n400_trf_per_subject_forward.csv").open()))
    order = [("biosemi128", "Lab cap\n128"), ("cyton_daisy16", "Starter\n16"),
             ("unicorn8", "Starter\n8 (Unicorn)"), ("cyton8", "Starter\n8 (Cyton)")]
    data = {m: [float(r["kernel_300_500ms"]) for r in rows if r["montage"] == m and r["feature"] == "surprisal"]
            for m, _ in order}

    fig = plt.figure(figsize=(13, 5.4))
    badge(fig, "Measured")
    ax = fig.add_axes([0.06, 0.13, 0.42, 0.66])
    t = np.linspace(-0.1, 0.9, 400)
    expected = -0.15 * np.exp(-((t - 0.1) / 0.04) ** 2) + 0.3 * np.exp(-((t - 0.2) / 0.06) ** 2)
    surprise = expected - 0.9 * np.exp(-((t - 0.4) / 0.08) ** 2)
    ax.axvspan(0.3, 0.5, color="#fff1e5", zorder=0)
    ax.plot(t, expected, color=MUTED, lw=2, label="expected word")
    ax.plot(t, surprise, color="#cf222e", lw=2.6, label="surprising word")
    ax.axhline(0, color="#d0d7de", lw=1)
    ax.set_xlabel("seconds after the word starts")
    ax.set_yticks([])
    ax.set_ylabel("brain response (sketch)")
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.47, -0.8, "N400", ha="left", color="#cf222e", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("The shape (textbook sketch)", fontsize=11, color=MUTED, loc="left")

    ax2 = fig.add_axes([0.56, 0.13, 0.41, 0.66])
    rng = np.random.default_rng(SEED)
    for i, (m, label) in enumerate(order):
        v = np.array(data[m])
        ax2.scatter(i + rng.uniform(-0.12, 0.12, v.size), v * 1000, s=26, alpha=0.75,
                    c=np.where(v < 0, "#cf222e", "#8c959f"), edgecolors="white", linewidths=0.5)
        ax2.plot([i - 0.25, i + 0.25], [v.mean() * 1000] * 2, color=INK, lw=2.5)
        ax2.text(i, 10.5, f"dipped for {(v < 0).sum()} of {v.size}", ha="center", fontsize=9, color=MUTED)
    ax2.axhline(0, color="#d0d7de", lw=1)
    ax2.set_xticks(range(len(order)), [lb for _, lb in order])
    ax2.set_ylabel("dip size at 0.3-0.5 s (each dot = one listener)")
    ax2.set_ylim(-30, 13)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("What we measured (red = dipped)", fontsize=11, color=MUTED, loc="left")
    save(fig, "06_words_in_brainwaves")


def fig_meaning_map() -> None:
    base = os.environ.get("SEMMN_BRODERICK")
    cache = Path(base) / "cache" / "lm_local.npz" if base else None
    if not cache or not cache.exists():
        print("skip 07_meaning_map: set SEMMN_BRODERICK to the extracted data folder with cache/lm_local.npz")
        return
    d = np.load(cache, allow_pickle=True)
    emb, surpr = d["embedding"].astype(float), d["surprisal"].astype(float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)  # compare directions (meaning), not lengths
    emb = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(emb, full_matrices=False)
    xyz = emb @ vt[:3].T
    xy = xyz[:, :2]
    med = np.median(xy, axis=0)
    mad = np.median(np.abs(xy - med), axis=0) * 1.4826
    lo, hi = med - 3.2 * mad, med + 3.2 * mad
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)  # drop the few far outliers so the shape is visible
    rank = surpr.argsort().argsort() / (len(surpr) - 1)  # colour by rank so the scale is even

    fig = plt.figure(figsize=(13, 6.2))
    badge(fig, "Measured")
    ax = fig.add_axes([0.04, 0.06, 0.54, 0.78])
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(xy[keep].T, bw_method=0.25)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, dens, levels=10, cmap="Blues", alpha=0.35)
    ax.contour(gx, gy, dens, levels=10, colors="#8c959f", linewidths=0.4, alpha=0.6)
    order = np.argsort(rank[keep])
    pts, col = xy[keep][order], rank[keep][order]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=col, cmap="plasma", s=9, alpha=0.8, linewidths=0, vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1])
    cax = fig.add_axes([0.6, 0.22, 0.012, 0.45])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([])
    cax.text(0.5, 1.03, "surprising", transform=cax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    cax.text(0.5, -0.06, "expected", transform=cax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    save(fig, "07_meaning_map")


def _summary_gain() -> dict[str, tuple[float, str]]:
    text = (EEG / "summary.md").read_text()
    block = text.split("### forward: word-feature gain", 1)[1].split("###", 1)[0]
    out = {}
    for m in re.finditer(r"^\| (\w+) \| \+?([-\d.]+) \| (\d+/\d+) \|", block, re.M):
        out[m.group(1)] = (float(m.group(2)), m.group(3))
    return out


def fig_scoreboard() -> None:
    gain = _summary_gain()
    fig = plt.figure(figsize=(13, 5.6))
    badge(fig, "Measured")

    ax = fig.add_axes([0.17, 0.2, 0.3, 0.56])
    order = [("biosemi128", "Lab cap 128"), ("cyton_daisy16", "Starter 16"), ("unicorn8", "Starter 8 (Unicorn)"),
             ("cyton8", "Starter 8 (Cyton)")]
    vals = [gain[m][0] for m, _ in order]
    ax.barh(range(len(order)), vals, color=["#57606a", "#1a7f37", "#1a7f37", "#1a7f37"], height=0.6)
    for i, (m, _) in enumerate(order):
        ax.text(vals[i] + 0.0001, i, f"  better for {gain[m][1]} listeners", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(order)), [lb for _, lb in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.9)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title("Works: knowing the words helps predict the brain waves", fontsize=11, loc="left")
    ax.text(0, -0.14, "Small but real (about +0.004 in correlation),\nand cheap headsets keep most of it.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="top")

    ax2 = fig.add_axes([0.66, 0.2, 0.3, 0.56])
    rows = [("Simple baseline (MLP)", 0.919, "#57606a"), ("3D-SEMMN", 0.881, "#0969da"),
            ("3D-SEMMN, wiring cut", 0.112, "#cf222e")]
    ax2.barh(range(3), [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax2.text(r[1] + 0.01, i, f"{r[1]:.0%}", va="center", fontsize=10)
    ax2.set_yticks(range(3), [r[0] for r in rows])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.12)
    ax2.set_xticks([])
    ax2.spines[["top", "right", "bottom"]].set_visible(False)
    ax2.set_title("Not yet: the brain-shaped model, on a toy digit task", fontsize=11, loc="left")
    ax2.text(0, -0.14, "It nearly matches a plain network, and its wiring matters.\nIt hasn't been run on EEG yet.",
             transform=ax2.transAxes, fontsize=9, color=MUTED, va="top")
    save(fig, "08_scoreboard")


FIGURES = {
    "idea": (fig_big_idea, False), "brain": (fig_brain, True), "wiring": (fig_wiring, True),
    "electrodes": (fig_electrodes, True), "headset": (fig_headset, False), "n400": (fig_n400, False),
    "meaning": (fig_meaning_map, False), "scoreboard": (fig_scoreboard, False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=list(FIGURES))
    args = ap.parse_args()
    names = args.only or list(FIGURES)
    atlas = load_atlas() if any(FIGURES[n][1] for n in names) else None
    for n in names:
        fn, needs_atlas = FIGURES[n]
        fn(atlas) if needs_atlas else fn()


if __name__ == "__main__":
    main()
